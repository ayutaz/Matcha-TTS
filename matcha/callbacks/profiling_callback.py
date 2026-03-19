import time

import torch
from lightning import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only


class ProfilingCallback(Callback):
    """Profiles training step timing to identify bottlenecks."""

    def __init__(self, profile_steps=100, skip_first=10, log_interval=50):
        self.profile_steps = profile_steps
        self.skip_first = skip_first
        self.log_interval = log_interval
        self.step_times = []
        self.batch_start_time = None
        self.epoch_start_time = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.global_step >= self.skip_first:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.batch_start_time = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.batch_start_time is not None and trainer.global_step >= self.skip_first:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - self.batch_start_time
            self.step_times.append(elapsed)

            if len(self.step_times) % self.log_interval == 0:
                self._log_stats(trainer)

            if len(self.step_times) >= self.profile_steps:
                self._log_final_report(trainer)
                self.step_times = []  # Reset

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.epoch_start_time is not None:
            elapsed = time.perf_counter() - self.epoch_start_time
            rank_zero_info(f"Epoch {trainer.current_epoch} completed in {elapsed:.1f}s")
            if torch.cuda.is_available():
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
                rank_zero_info(f"  Peak GPU memory: {mem_gb:.2f} GB")

    @rank_zero_only
    def _log_stats(self, trainer):
        import statistics

        recent = self.step_times[-self.log_interval :]
        mean_ms = statistics.mean(recent) * 1000
        rank_zero_info(
            f"[Profiling] Steps {len(self.step_times)}: {mean_ms:.1f} ms/step ({1000/mean_ms:.1f} steps/s)"
        )

    @rank_zero_only
    def _log_final_report(self, trainer):
        import statistics

        times = self.step_times
        mean_ms = statistics.mean(times) * 1000
        median_ms = statistics.median(times) * 1000
        std_ms = statistics.stdev(times) * 1000 if len(times) > 1 else 0

        rank_zero_info("=" * 60)
        rank_zero_info("PROFILING REPORT")
        rank_zero_info(f"  Steps profiled: {len(times)}")
        rank_zero_info(f"  Mean:   {mean_ms:.1f} ms/step ({1000/mean_ms:.1f} steps/s)")
        rank_zero_info(f"  Median: {median_ms:.1f} ms/step")
        rank_zero_info(f"  Stdev:  {std_ms:.1f} ms")
        rank_zero_info(f"  Min:    {min(times)*1000:.1f} ms")
        rank_zero_info(f"  Max:    {max(times)*1000:.1f} ms")
        if torch.cuda.is_available():
            rank_zero_info(f"  GPU Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB peak")
        rank_zero_info("=" * 60)
