"""
训练循环骨架：train_epoch、validate、fit。
支持孪生网络成对前向与对比损失。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

_log = logging.getLogger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Trainer:
    """
    训练器：执行 train_epoch、validate、fit，保存最佳模型。
    需配合 step_fn 处理成对数据的孪生前向。
    checkpoint_meta: 非空时保存 torch 文件为 {state_dict, meta}（供 MultiModalFusion 等读取）；否则保存裸 state_dict。
    """

    def __init__(
        self,
        model: Any,
        train_loader: Any,
        val_loader: Any,
        loss_fn: Any,
        optimizer: Any,
        device: Any,
        save_path: str,
        step_fn: Optional[Callable[[Any, Any, Any], Any]] = None,
        similarity_threshold: float = 0.5,
        tb_writer: Optional[Any] = None,
        checkpoint_meta: Optional[Dict[str, Any]] = None,
        use_amp: bool = False,
        accumulation_steps: int = 1,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for Trainer")
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.step_fn = step_fn
        self.similarity_threshold = similarity_threshold
        self.tb_writer = tb_writer
        self.checkpoint_meta = checkpoint_meta
        self.best_val_loss: Optional[float] = None
        self.use_amp = use_amp and hasattr(device, "type") and device.type == "cuda"
        self.accumulation_steps = max(1, int(accumulation_steps))
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

    def _default_step(self, batch: Dict[str, Any], model: Any, loss_fn: Any) -> "torch.Tensor":
        """默认步进：需由外部注入 vocab 与 tensorize。此处为占位。"""
        raise NotImplementedError("Provide step_fn to Trainer")

    def _run_epoch(
        self,
        loader: Any,
        training: bool,
        *,
        epoch: int = 0,
        num_epochs: int = 1,
        phase: str = "train",
        progress_bar: bool = False,
        log_batches_every: int = 20,
    ) -> tuple[float, float]:
        """执行一个 epoch，返回 (avg_loss, accuracy)。"""
        self.model.train() if training else self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        iterator: Any = loader
        pbar = None
        tqdm_mod = None
        if progress_bar:
            try:
                from tqdm import tqdm as tqdm_mod  # type: ignore
            except ImportError:
                tqdm_mod = None
            if tqdm_mod is not None:
                desc = f"{phase} {epoch + 1}/{num_epochs}"
                pbar = tqdm_mod(loader, desc=desc, leave=True, unit="batch", dynamic_ncols=True)
                iterator = pbar

        use_periodic_log = progress_bar and tqdm_mod is None and log_batches_every > 0
        since_log = 0
        window_loss = 0.0
        window_count = 0

        # AMP autocast context (nullcontext when AMP off or CPU)
        if self.use_amp:
            autocast_ctx = torch.amp.autocast("cuda")
        else:
            from contextlib import nullcontext as _nullcontext

            autocast_ctx = _nullcontext()

        try:
            n_batches = len(loader) if hasattr(loader, "__len__") else None
            if epoch == 0 and phase == "train":
                _log.info(
                    "Trainer: 首个 epoch 的第一个 batch 可能较慢（DataLoader 取数、懒读预计算特征、"
                    "首次 GPU 前向/编译）；进度条在首个 batch 完成前会停留在起始位置。"
                )
            t_epoch_loop = time.perf_counter()
            log_first_batch_detail = epoch == 0
            oom_skipped = 0
            for batch_idx, batch in enumerate(iterator):
                if batch_idx == 0 and log_first_batch_detail:
                    t_after_data = time.perf_counter()
                    _log.info(
                        "Trainer: [%s] 首个 batch 数据已就绪（DataLoader+collate）耗时 %.2fs",
                        phase,
                        t_after_data - t_epoch_loop,
                    )
                t_before_step = time.perf_counter()
                try:
                    with autocast_ctx:
                        if self.step_fn is None:
                            loss = self._default_step(batch, self.model, self.loss_fn)
                            correct = 0
                            count = 1
                        else:
                            loss, correct, count = self.step_fn(batch, self.model, self.loss_fn)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() and training:
                        oom_skipped += 1
                        _log.warning(
                            "Trainer: [train] batch %d CUDA OOM，跳过本 batch（累计跳过 %d 次）。"
                            "建议减小 --batch-size 或 --max-seq-len。",
                            batch_idx, oom_skipped,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        import gc as _gc
                        _gc.collect()
                        continue
                    raise
                if batch_idx == 0 and log_first_batch_detail:
                    t_after_step = time.perf_counter()
                    _log.info(
                        "Trainer: [%s] 首个 batch step_fn 完成耗时 %.2fs count=%s",
                        phase,
                        t_after_step - t_before_step,
                        count,
                    )
                if count <= 0:
                    continue
                loss_item = loss.item()
                total_loss += loss_item * count
                total_correct += correct
                total_count += count
                if training:
                    # ── Gradient accumulation（修复版）──
                    # 修复前：在 accumulation 窗口首 batch 之前就 zero_grad，逻辑冗余。
                    # 修复后：只在首次 batch 和每次 step 后 zero_grad，确保梯度正确累积。
                    if batch_idx == 0:
                        self.optimizer.zero_grad()
                    # Scale loss by accumulation steps for correct gradient averaging
                    scaled_loss = loss / self.accumulation_steps
                    if self.scaler is not None:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    # Step at accumulation boundary or last batch
                    is_last_batch = (batch_idx + 1 == n_batches) if n_batches is not None else False
                    if (batch_idx + 1) % self.accumulation_steps == 0 or is_last_batch:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        # 梯度裁剪：防止梯度爆炸导致训练不稳定和 OOM
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        if self.scaler is not None:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad()

                if batch_idx == 0 and training and count > 0 and log_first_batch_detail:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_after_backward = time.perf_counter()
                    _log.info(
                        "Trainer: [train] 首个 batch 反传与 optimizer.step 后总耗时（自 step_fn 起）%.2fs",
                        t_after_backward - t_before_step,
                    )

                if pbar is not None:
                    pbar.set_postfix(loss=f"{loss_item:.4f}", refresh=False)
                elif use_periodic_log:
                    window_loss += loss_item * count
                    window_count += count
                    since_log += 1
                    if since_log >= log_batches_every:
                        wavg = window_loss / window_count if window_count else 0.0
                        total_s = f"/{n_batches}" if n_batches is not None else ""
                        _log.info(
                            "  [%s] batch %s%s  window_avg_loss=%.4f",
                            phase,
                            batch_idx + 1,
                            total_s,
                            wavg,
                        )
                        since_log = 0
                        window_loss = 0.0
                        window_count = 0
        finally:
            if pbar is not None and hasattr(pbar, "close"):
                pbar.close()

        if oom_skipped > 0:
            _log.warning(
                "Trainer: [%s epoch %d] 累计跳过 %d 个 batch（CUDA OOM）。"
                "若频繁出现请减小 --batch-size 或 --max-seq-len。",
                phase, epoch, oom_skipped,
            )

        avg_loss = total_loss / total_count if total_count else 0.0
        acc = total_correct / total_count if total_count else 0.0
        return avg_loss, acc

    def train_epoch(self, **run_kw: Any) -> float:
        """执行一个训练 epoch，返回平均损失。"""
        avg_loss, _ = self._run_epoch(self.train_loader, training=True, **run_kw)
        return avg_loss

    def validate(self, **run_kw: Any) -> tuple[float, float]:
        """在验证集上计算损失和准确率，返回 (val_loss, val_acc)。"""
        with torch.no_grad():
            return self._run_epoch(self.val_loader, training=False, **run_kw)

    def fit(
        self,
        num_epochs: int,
        on_epoch_end: Optional[Callable[[int, float, float, float], None]] = None,
        *,
        on_epoch_begin: Optional[Callable[[int], None]] = None,
        progress_bar: bool = False,
        log_batches_every: int = 20,
        cleanup_every_epoch: bool = False,
    ) -> None:
        """循环训练与验证，保存最佳权重。
        on_epoch_end: 可选回调 (epoch_idx, train_loss, val_loss, val_acc)，用于外部日志（如 W&B）。
        on_epoch_begin: 可选回调 (epoch_idx)，在每个 epoch 训练开始前调用（如刷新固定样本对）。

        cleanup_every_epoch 行为变更（修复 epoch 间缓存清空导致训练变慢/异常）：
          - 当数据集使用预计算特征（有 clear_runtime_cache 且带 _precomputed_lazy_index
            或 _precomputed_features）时，**仅清 lsir_raw_cache**，保留 memory_cache。
            原因：memory_cache 是 JSONL 随机读的缓存层，清空后每个 __getitem__ 都要
            fseek+json.loads，比 dict 查找慢 100-1000x，导致后续 epoch 训练极慢且
            可能因 I/O 瓶颈导致 DataLoader 超时。
          - 仅当数据集做动态提取（无预计算特征）时，才清空全部缓存（含 memory_cache），
            因为此时 lsir_raw 才是真正的内存大头。
          - gc.collect() 和 cuda.empty_cache() 始终执行。
        """
        import os

        d = os.path.dirname(self.save_path)
        if d:
            os.makedirs(d, exist_ok=True)

        run_kw: Dict[str, Any] = {
            "num_epochs": num_epochs,
            "progress_bar": progress_bar,
            "log_batches_every": max(1, int(log_batches_every)),
        }
        if progress_bar:
            try:
                n_tr = len(self.train_loader)
                n_va = len(self.val_loader)
                _log.info(
                    "[Trainer] 每 epoch: 训练 %s batch | 验证 %s batch",
                    n_tr,
                    n_va,
                )
            except TypeError:
                _log.info("[Trainer] 进度条已启用（DataLoader 长度未知，无 batch 总数）")
            try:
                import tqdm  # noqa: F401
            except ImportError:
                _log.warning(
                    "[Trainer] 未安装 tqdm，将每 %d 个 batch 打印一行进度；"
                    "可 pip install tqdm 获得进度条。",
                    run_kw["log_batches_every"],
                )

        def _dataset_uses_precomputed(loader: Any) -> bool:
            """检测数据集是否使用预计算特征（JSONL / JSON map），决定清理策略。"""
            ds = getattr(loader, "dataset", None)
            visited = set()
            while ds is not None and id(ds) not in visited:
                visited.add(id(ds))
                # 有 lazy index 或 precomputed map → 使用预计算特征
                if getattr(ds, "_precomputed_lazy_index", None) is not None:
                    return True
                if getattr(ds, "_precomputed_features", None):
                    return True
                ds = getattr(ds, "dataset", None)
            return False

        def _selective_clear_runtime_cache(loader: Any, *, clear_memory: bool) -> None:
            """按策略选择性清理运行期缓存。"""
            ds = getattr(loader, "dataset", None)
            visited = set()
            while ds is not None and id(ds) not in visited:
                visited.add(id(ds))
                clear_fn = getattr(ds, "clear_runtime_cache", None)
                if callable(clear_fn):
                    try:
                        clear_fn(clear_memory=clear_memory, clear_lsir=True)
                    except TypeError:
                        # 兼容旧版 clear_runtime_cache() 无参数签名
                        try:
                            clear_fn()
                        except Exception:
                            pass
                    except Exception:
                        pass
                ds = getattr(ds, "dataset", None)

        train_uses_precomputed = _dataset_uses_precomputed(self.train_loader)
        val_uses_precomputed = _dataset_uses_precomputed(self.val_loader)

        if cleanup_every_epoch:
            if train_uses_precomputed or val_uses_precomputed:
                _log.info(
                    "[Trainer] epoch cleanup: 保留 memory_cache（预计算特征），仅清理 lsir_raw + gc + cuda"
                )
            else:
                _log.info(
                    "[Trainer] epoch cleanup: 清理全部缓存（动态提取模式）+ gc + cuda"
                )

        for epoch in range(num_epochs):
            run_kw["epoch"] = epoch
            if on_epoch_begin is not None:
                on_epoch_begin(epoch)
            train_loss = self.train_epoch(phase="train", **run_kw)
            val_loss, val_acc = self.validate(phase="val", **run_kw)
            _log.info(
                "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
                epoch + 1,
                num_epochs,
                train_loss,
                val_loss,
                val_acc,
            )
            if self.tb_writer:
                self.tb_writer.add_scalar("loss/train", train_loss, epoch)
                self.tb_writer.add_scalar("loss/val", val_loss, epoch)
                self.tb_writer.add_scalar("acc/val", val_acc, epoch)
            if on_epoch_end is not None:
                on_epoch_end(epoch, train_loss, val_loss, val_acc)
            if self.best_val_loss is None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if self.checkpoint_meta is not None:
                    torch.save(
                        {"state_dict": self.model.state_dict(), "meta": dict(self.checkpoint_meta)},
                        self.save_path,
                    )
                else:
                    torch.save(self.model.state_dict(), self.save_path)
            if cleanup_every_epoch:
                import gc

                # 预计算特征路径：仅清 lsir_raw_cache，保留 memory_cache（避免 JSONL 随机读瓶颈）
                # 动态提取路径：清全部缓存（lsir_raw 才是大头）
                _selective_clear_runtime_cache(
                    self.train_loader, clear_memory=not train_uses_precomputed
                )
                _selective_clear_runtime_cache(
                    self.val_loader, clear_memory=not val_uses_precomputed
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
