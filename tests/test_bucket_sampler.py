"""Tests for BucketBatchSampler and DistributedBucketBatchSampler in precomputed_datamodule."""

import random

import pytest

from matcha.data.precomputed_datamodule import (
    BucketBatchSampler,
    DistributedBucketBatchSampler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_SAMPLES = 1000
BATCH_SIZE = 32


def _make_file_sizes(n=NUM_SAMPLES, seed=42):
    """Generate dummy file sizes with realistic variation (proxy for mel length).

    Sizes range from ~5000 to ~50000, roughly mimicking .pt file sizes for
    mel spectrograms of varying utterance lengths.
    """
    rng = random.Random(seed)
    return [rng.randint(5000, 50000) for _ in range(n)]


def _collect_all_indices(sampler):
    """Iterate through the sampler and return a flat list of all yielded indices."""
    all_indices = []
    for batch in sampler:
        all_indices.extend(batch)
    return all_indices


def _compute_batch_variance(batches, file_sizes):
    """Compute the average within-batch variance of file sizes.

    Lower variance means items within each batch are more similar in length,
    resulting in less padding waste.
    """
    variances = []
    for batch in batches:
        if len(batch) < 2:
            continue
        sizes = [file_sizes[i] for i in batch]
        mean = sum(sizes) / len(sizes)
        var = sum((s - mean) ** 2 for s in sizes) / len(sizes)
        variances.append(var)
    return sum(variances) / len(variances) if variances else 0.0


# ---------------------------------------------------------------------------
# BucketBatchSampler tests
# ---------------------------------------------------------------------------


class TestBucketBatchSampler:
    def test_default_num_buckets_is_20(self):
        """Default num_buckets should be 20."""
        file_sizes = _make_file_sizes()
        sampler = BucketBatchSampler(file_sizes, batch_size=BATCH_SIZE)
        # 20 buckets for 1000 samples => bucket_size = 50, so 20 buckets
        assert len(sampler.buckets) == 20

    def test_custom_num_buckets(self):
        """num_buckets parameter should control the number of buckets created."""
        file_sizes = _make_file_sizes()
        sampler = BucketBatchSampler(file_sizes, batch_size=BATCH_SIZE, num_buckets=5)
        assert len(sampler.buckets) == 5

    def test_all_samples_covered_no_drop_last(self):
        """All samples should appear exactly once when drop_last=False."""
        file_sizes = _make_file_sizes()
        sampler = BucketBatchSampler(file_sizes, batch_size=BATCH_SIZE, num_buckets=20, drop_last=False)
        all_indices = _collect_all_indices(sampler)
        assert sorted(all_indices) == list(range(NUM_SAMPLES))

    def test_all_samples_subset_drop_last(self):
        """With drop_last=True, yielded indices should be a subset of all indices,
        and every batch should be exactly batch_size."""
        file_sizes = _make_file_sizes()
        sampler = BucketBatchSampler(file_sizes, batch_size=BATCH_SIZE, num_buckets=20, drop_last=True)
        batches = list(sampler)
        all_indices = []
        for batch in batches:
            assert len(batch) == BATCH_SIZE
            all_indices.extend(batch)
        # All yielded indices should be valid
        assert all(0 <= i < NUM_SAMPLES for i in all_indices)
        # No duplicates
        assert len(all_indices) == len(set(all_indices))

    def test_more_buckets_reduce_within_batch_variance(self):
        """Increasing num_buckets should reduce within-batch file size variance,
        meaning less padding waste."""
        file_sizes = _make_file_sizes()

        sampler_10 = BucketBatchSampler(file_sizes, batch_size=BATCH_SIZE, num_buckets=10, drop_last=False, seed=0)
        batches_10 = list(sampler_10)
        var_10 = _compute_batch_variance(batches_10, file_sizes)

        sampler_20 = BucketBatchSampler(file_sizes, batch_size=BATCH_SIZE, num_buckets=20, drop_last=False, seed=0)
        batches_20 = list(sampler_20)
        var_20 = _compute_batch_variance(batches_20, file_sizes)

        # 20 buckets should have less or equal within-batch variance than 10 buckets
        assert var_20 <= var_10, (
            f"Expected 20-bucket variance ({var_20:.1f}) <= 10-bucket variance ({var_10:.1f})"
        )

    def test_len_is_reasonable_estimate_drop_last_false(self):
        """__len__ should be a reasonable estimate of the batch count (drop_last=False).

        The __len__ computes total_samples / batch_size (ceiling), which is an
        approximation because samples are split across buckets. The actual batch
        count may differ slightly due to per-bucket remainder batches.
        """
        file_sizes = _make_file_sizes()
        sampler = BucketBatchSampler(file_sizes, batch_size=BATCH_SIZE, num_buckets=20, drop_last=False)
        batches = list(sampler)
        reported_len = len(sampler)
        actual_len = len(batches)
        # __len__ is based on total/batch_size; actual count may be higher due
        # to per-bucket remainders becoming separate batches.  Check they are
        # within a reasonable range of each other.
        assert actual_len >= reported_len, "Actual batches should be >= __len__ estimate"
        # The excess should be at most num_buckets (one remainder per bucket)
        assert actual_len - reported_len <= 20

    def test_len_is_reasonable_estimate_drop_last_true(self):
        """__len__ should be a reasonable estimate of the batch count (drop_last=True).

        With drop_last=True, __len__ computes total // batch_size but actual
        count may be lower because per-bucket remainders are discarded.
        """
        file_sizes = _make_file_sizes()
        sampler = BucketBatchSampler(file_sizes, batch_size=BATCH_SIZE, num_buckets=20, drop_last=True)
        batches = list(sampler)
        reported_len = len(sampler)
        actual_len = len(batches)
        # With drop_last, actual should be <= reported (remainders dropped per bucket)
        assert actual_len <= reported_len, "Actual batches should be <= __len__ estimate"
        # The deficit should be at most num_buckets (one dropped remainder per bucket)
        assert reported_len - actual_len <= 20


# ---------------------------------------------------------------------------
# DistributedBucketBatchSampler tests
# ---------------------------------------------------------------------------


class TestDistributedBucketBatchSampler:
    def test_default_num_buckets_is_20(self):
        """Default num_buckets should be 20."""
        file_sizes = _make_file_sizes()
        sampler = DistributedBucketBatchSampler(
            file_sizes, batch_size=BATCH_SIZE, num_replicas=2, rank=0
        )
        assert sampler.num_buckets == 20

    def test_custom_num_buckets(self):
        """num_buckets parameter should be stored and used."""
        file_sizes = _make_file_sizes()
        sampler = DistributedBucketBatchSampler(
            file_sizes, batch_size=BATCH_SIZE, num_replicas=2, rank=0, num_buckets=15
        )
        assert sampler.num_buckets == 15

    def test_all_samples_covered_across_ranks(self):
        """Union of indices across all ranks should cover every sample (with possible
        padding duplicates for even division)."""
        file_sizes = _make_file_sizes()
        num_replicas = 4
        all_indices = []
        for rank in range(num_replicas):
            sampler = DistributedBucketBatchSampler(
                file_sizes,
                batch_size=BATCH_SIZE,
                num_replicas=num_replicas,
                rank=rank,
                num_buckets=20,
                drop_last=False,
            )
            all_indices.extend(_collect_all_indices(sampler))

        # Every original index should appear at least once
        unique_indices = set(i % NUM_SAMPLES for i in all_indices)
        assert unique_indices == set(range(NUM_SAMPLES))

    def test_drop_last_batches_are_full(self):
        """With drop_last=True, every batch should have exactly batch_size items."""
        file_sizes = _make_file_sizes()
        sampler = DistributedBucketBatchSampler(
            file_sizes,
            batch_size=BATCH_SIZE,
            num_replicas=2,
            rank=0,
            num_buckets=20,
            drop_last=True,
        )
        for batch in sampler:
            assert len(batch) == BATCH_SIZE

    def test_ranks_get_disjoint_indices(self):
        """Different ranks should receive disjoint sets of indices (before padding)."""
        file_sizes = _make_file_sizes()
        num_replicas = 2
        indices_per_rank = []
        for rank in range(num_replicas):
            sampler = DistributedBucketBatchSampler(
                file_sizes,
                batch_size=BATCH_SIZE,
                num_replicas=num_replicas,
                rank=rank,
                num_buckets=20,
                drop_last=False,
            )
            indices_per_rank.append(set(_collect_all_indices(sampler)))

        # Check that ranks have disjoint index sets
        overlap = indices_per_rank[0] & indices_per_rank[1]
        # Some overlap is possible due to padding (wrapping extra indices to make
        # evenly divisible), but the overlap should be minimal
        max_padding = num_replicas - 1
        assert len(overlap) <= max_padding, (
            f"Ranks share {len(overlap)} indices, expected at most {max_padding}"
        )

    def test_set_epoch_changes_order(self):
        """Calling set_epoch should change the iteration order."""
        file_sizes = _make_file_sizes()
        sampler = DistributedBucketBatchSampler(
            file_sizes,
            batch_size=BATCH_SIZE,
            num_replicas=1,
            rank=0,
            num_buckets=20,
            drop_last=False,
        )

        sampler.set_epoch(0)
        batches_epoch0 = [tuple(b) for b in sampler]

        sampler.set_epoch(1)
        batches_epoch1 = [tuple(b) for b in sampler]

        # The batch order should differ between epochs
        assert batches_epoch0 != batches_epoch1

    def test_more_buckets_reduce_within_batch_variance(self):
        """Increasing num_buckets should reduce within-batch file size variance
        for the distributed sampler as well."""
        file_sizes = _make_file_sizes()

        sampler_10 = DistributedBucketBatchSampler(
            file_sizes,
            batch_size=BATCH_SIZE,
            num_replicas=1,
            rank=0,
            num_buckets=10,
            drop_last=False,
        )
        batches_10 = list(sampler_10)
        var_10 = _compute_batch_variance(batches_10, file_sizes)

        sampler_20 = DistributedBucketBatchSampler(
            file_sizes,
            batch_size=BATCH_SIZE,
            num_replicas=1,
            rank=0,
            num_buckets=20,
            drop_last=False,
        )
        batches_20 = list(sampler_20)
        var_20 = _compute_batch_variance(batches_20, file_sizes)

        assert var_20 <= var_10, (
            f"Expected 20-bucket variance ({var_20:.1f}) <= 10-bucket variance ({var_10:.1f})"
        )

    def test_len_is_reasonable_estimate_drop_last_true(self):
        """__len__ should be a reasonable estimate of the batch count (drop_last=True).

        The __len__ computes num_samples // batch_size, but actual count may
        differ because samples are split across buckets and per-bucket remainders
        are dropped.
        """
        file_sizes = _make_file_sizes()
        sampler = DistributedBucketBatchSampler(
            file_sizes,
            batch_size=BATCH_SIZE,
            num_replicas=2,
            rank=0,
            num_buckets=20,
            drop_last=True,
        )
        batches = list(sampler)
        reported_len = len(sampler)
        actual_len = len(batches)
        # With drop_last, actual should be <= reported
        assert actual_len <= reported_len, "Actual batches should be <= __len__ estimate"
        # The deficit should be at most num_buckets
        assert reported_len - actual_len <= 20
