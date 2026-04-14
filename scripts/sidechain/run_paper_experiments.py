import json, os, sys, time
import numpy as np
import faiss, ijson, sqlite3
from tqdm import tqdm
sys.path.insert(0, "src")
from features.baselines.safe import SafeEmbedder
from matcher.rerank import RerankModel
LIB_EMB = "data/two_stage/library_safe_embeddings_full20.json"
LIB_DB = "data/two_stage/library_features.db"
GT_PATH = "data/two_stage/ground_truth.json"
QF_PATH = "data/two_stage/query_features.json"
SAFE_MODEL = "output/safe_full_20ep.pt"
RERANK_MODEL = "output/best_model_library_only.pth"
OUT_DIR = "output/benchmarks/paper"
os.makedirs(OUT_DIR, exist_ok=True)
def build_index(path):
    ids, vecs = [], []
    with open(path, "rb") as f:
        for obj in ijson.items(f, "functions.item"):
            ids.append(obj["function_id"])
            vecs.append([float(x) for x in obj["vector"]])
    mat = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.where(norms > 0, norms, 1)
    idx = faiss.IndexFlatIP(mat.shape[1])
    idx.add(mat)
    return idx, ids
def get_q_embs(gt, model_path):
    emb = SafeEmbedder(model_path=model_path, device="cuda", prefer_cuda=True)
    qmm = {}
    with open(QF_PATH, "rb") as f:
        for qid, val in ijson.kvitems(f, ""):
            if qid in gt and isinstance(val, dict):
                qmm[qid] = val
    qids = list(qmm.keys())
    vecs = emb.embed_many([qmm[q] for q in qids], batch_size=256)
    return {q: np.array(v, dtype=np.float32) for q, v in zip(qids, vecs)}, qmm
def compute_metrics(ranked, gt, k):
    n = len(ranked)
    if n == 0: return 0.0, 0.0, 0.0
    rs = ps = ms = 0.0
    for qid, rk in ranked.items():
        pos = set(gt.get(qid, []))
        top = rk[:k]
        h = sum(1 for c in top if c in pos)
        rs += 1.0 if h > 0 else 0.0
        ps += h / k
        for rank, c in enumerate(top, 1):
            if c in pos:
                ms += 1.0 / rank
                break
    return rs / n, ps / n, ms / n
def run(name, idx, lid, gt_f, qemb, qmm, rerank=None, ck=100, maxq=None):
    print()
    print("=" * 60)
    print("Experiment: " + name)
    print("=" * 60)
    qids = list(qemb.keys())
    if maxq:
        qids = qids[:maxq]
    conn = sqlite3.connect(LIB_DB)
    cur = conn.cursor()
    t0 = time.time()
    ranked = {}
    ch = tied = changed = 0
    for qid in tqdm(qids, desc=name):
        tgt = set(gt_f.get(qid, []))
        qv = qemb[qid]
        qn = np.linalg.norm(qv)
        if qn > 0: qv = qv / qn
        sc, ind = idx.search(qv.reshape(1, -1), ck)
        cids = [lid[j] for j in ind[0] if j != -1]
        if set(cids) & tgt: ch += 1
        if rerank:
            cf = []
            for cid in cids:
                cur.execute("SELECT features_json FROM features WHERE function_id=?", (cid,))
                row = cur.fetchone()
                if row:
                    mm = json.loads(row[0])
                    if mm: cf.append((cid, mm))
            qm = qmm.get(qid, {})
            if qm and cf:
                sc2 = rerank.score(qm, cf)
                ranked[qid] = [c for c, _ in sc2]
                if len(sc2) >= 2 and abs(sc2[0][1] - sc2[1][1]) < 1e-9: tied += 1
                if cids[:10] != [c for c, _ in sc2][:10]: changed += 1
            else: ranked[qid] = cids
        else: ranked[qid] = cids
    elapsed = time.time() - t0
    conn.close()
    n = len(qids)
    res = {}
    for k in [1, 5, 10, 20, 50]:
        r, p, m = compute_metrics(ranked, gt_f, k)
        res[k] = {"recall": r, "precision": p, "mrr": m}
    print()
    print("  K     Recall    Precision   MRR")
    print("  " + "-" * 40)
    for k in sorted(res.keys()):
        v = res[k]
        print("  @%-3d  %.4f    %.4f      %.4f" % (k, v["recall"], v["precision"], v["mrr"]))
    diag = {"coarse_hit": ch, "total": n, "time": elapsed}
    if rerank:
        diag["tied"] = tied
        diag["changed"] = changed
    print()
    print("  coarse_hit: %d/%d = %.4f" % (ch, n, ch / n if n else 0))
    if rerank:
        print("  tied: %d/%d = %.4f" % (tied, n, tied / n if n else 0))
        print("  order_changed: %d/%d = %.4f" % (changed, n, changed / n if n else 0))
    print("  time: %.1fs" % elapsed)
    return {"name": name, "metrics": {str(k): v for k, v in res.items()}, "diag": diag}
def main():
    MAXQ = 500
    print("Loading ground truth...")
    with open(GT_PATH) as f: gt = json.load(f)
    print("Building FAISS...")
    idx, lid = build_index(LIB_EMB)
    lid_set = set(lid)
    gt_f = {q: t for q, t in gt.items() if set(t) & lid_set}
    print("GT filtered: %d queries" % len(gt_f))
    print("Computing query embeddings...")
    qemb, qmm = get_q_embs(gt_f, SAFE_MODEL)
    print("Query embeddings: %d" % len(qemb))
    print("Loading rerank model...")
    rerank = RerankModel(model_path=RERANK_MODEL, device="cuda", prefer_cuda=True)
    all_res = []
    for ck in [50, 100, 200]:
        r = run("SAFE_only_k%d" % ck, idx, lid, gt_f, qemb, qmm, None, ck, MAXQ)
        all_res.append(r)
    for ck in [50, 100, 200]:
        r = run("SAFE+Rerank_k%d" % ck, idx, lid, gt_f, qemb, qmm, rerank, ck, MAXQ)
        all_res.append(r)
    out = os.path.join(OUT_DIR, "all_experiments.json")
    with open(out, "w") as f: json.dump(all_res, f, indent=2)
    print()
    print("Saved to " + out)
    print()
    print("=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print("%-25s %8s %8s %8s %8s %10s %8s" % ("Method", "R@1", "R@5", "R@10", "R@50", "CoarseHit", "Time"))
    print("-" * 90)
    for r in all_res:
        m = r["metrics"]
        d = r["diag"]
        n = d["total"]
        print("%-25s %8.4f %8.4f %8.4f %8.4f %10.4f %8.1fs" % (
            r["name"],
            m["1"]["recall"], m["5"]["recall"], m["10"]["recall"], m["50"]["recall"],
            d["coarse_hit"] / n if n else 0, d["time"]))
if __name__ == "__main__":
    main()
