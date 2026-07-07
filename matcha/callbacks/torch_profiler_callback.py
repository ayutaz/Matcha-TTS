import os
from pathlib import Path

import torch
from lightning import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only


class TorchProfilerCallback(Callback):
    """Kernel-level profiling (A-1) to locate the training bottleneck.

    Wraps ``torch.profiler`` around a short window of training steps and, on rank 0,
    prints a ``key_averages()`` table sorted by self CUDA time plus exports a Chrome
    trace. Reading the table tells you what the run is limited by:

    * **GEMM-bound**            -> ``aten::addmm`` / ``mm`` / ``bmm`` dominate self CUDA time.
                                   torch.compile (A-2) / cuDNN SDPA (A-3) give little headroom.
    * **kernel-launch-bound**   -> many tiny elementwise / LayerNorm kernels, CPU total >> CUDA
                                   total, gaps in the trace. **A-2 regional compile is the lever.**
    * **communication-bound**   -> ``ncclDevKernel_AllReduce*`` is a large slice. Look at C-1 / batch.
    * **data / collate-bound**  -> GPU idle at step boundaries, ``aten::copy_`` / H2D prominent.
                                   Look at B-1 (frame batching) / B-3 (pin_memory).

    The callback never signals ``should_stop`` so all DDP ranks stay in lockstep; bound the
    run with ``trainer.limit_train_batches`` instead (must be >= wait+warmup+active).
    Profiling is done on rank 0 only to keep the trace readable and cheap.
    """

    def __init__(
        self,
        wait: int = 5,
        warmup: int = 5,
        active: int = 20,
        row_limit: int = 30,
        trace_dir: str = "logs/profiler",
        record_shapes: bool = True,
    ):
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.row_limit = row_limit
        self.trace_dir = trace_dir
        self.record_shapes = record_shapes
        self._prof = None
        self._done = False
        # Steps consumed before the profiler's active window closes (one schedule cycle).
        self._total = wait + warmup + active

    @rank_zero_only
    def _build(self):
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._prof = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self.wait, warmup=self.warmup, active=self.active, repeat=1
            ),
            record_shapes=self.record_shapes,
            with_stack=False,
            profile_memory=False,
        )
        self._prof.start()
        rank_zero_info(
            f"[TorchProfiler] started: wait={self.wait} warmup={self.warmup} active={self.active} "
            f"-> profiling steps {self.wait + self.warmup}..{self._total} "
            f"(set trainer.limit_train_batches >= {self._total})"
        )

    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero and self._prof is None and not self._done:
            self._build()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only rank 0 holds a profiler object; other ranks are unaffected (DDP lockstep kept).
        if self._prof is None or self._done:
            return
        self._prof.step()
        if (batch_idx + 1) >= self._total:
            self._finish()

    @rank_zero_only
    def _finish(self):
        if self._prof is None or self._done:
            return
        self._prof.stop()
        self._done = True

        cuda_ok = torch.cuda.is_available()
        sort_key = "self_cuda_time_total" if cuda_ok else "self_cpu_time_total"
        # group_by_input_shape=False => a single readable table instead of one row per mel
        # length (variable-length training would otherwise shatter the aggregation).
        table = self._prof.key_averages(group_by_input_shape=False).table(
            sort_by=sort_key, row_limit=self.row_limit
        )
        rank_zero_info("=" * 80)
        rank_zero_info(f"TORCH PROFILER REPORT (sorted by {sort_key})")
        rank_zero_info(
            "Read: GEMM(addmm/mm/bmm) high => compute-bound; many tiny kernels + CPU>>CUDA "
            "=> launch-bound (A-2); ncclDevKernel_* high => comm-bound; copy_/idle gaps => data-bound."
        )
        rank_zero_info("\n" + table)
        rank_zero_info("=" * 80)

        Path(self.trace_dir).mkdir(parents=True, exist_ok=True)
        trace_path = os.path.join(self.trace_dir, "trace_rank0.json")
        try:
            self._prof.export_chrome_trace(trace_path)
            rank_zero_info(f"[TorchProfiler] chrome trace -> {trace_path} (open in chrome://tracing)")
        except Exception as exc:  # noqa: BLE001 - trace export must never crash training
            rank_zero_info(f"[TorchProfiler] chrome trace export failed: {exc}")
