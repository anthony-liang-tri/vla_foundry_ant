import json
from typing import Any, Dict, List

import numpy as np
import ray


@ray.remote
class StreamingDatasetStatistics:
    """Thread-safe memory-efficient streaming statistics computation."""

    def __init__(self, compute_stats: bool = True, max_samples_for_percentiles: int = 100000):
        self.compute_stats = compute_stats
        self.max_samples_for_percentiles = max_samples_for_percentiles
        if not compute_stats:
            return

        self.running_means = {}
        self.running_m2s = {}
        self.counts = {}
        self.mins = {}
        self.maxs = {}
        self.samples_for_percentiles = {}  # Store samples for percentile computation
        self.sample_mask_for_percentiles = {}  # Store masks for percentile computation

    def update(self, sample_lowdim: Dict[str, np.ndarray]):
        """Thread-safe update of statistics with new sample (Welford's online algorithm)."""
        if not self.compute_stats:
            return

        mask = sample_lowdim["mask"][..., None]
        for key, data in sample_lowdim.items():
            if not np.issubdtype(data.dtype, np.number):
                continue
            if data.ndim == 2:
                # Add a batch dimension, assume there is only time and channel dimensions
                data = data[None, ...]
                mask = mask[None, ...]
            elif data.ndim != 3:
                raise ValueError(f"Data must have 2 or 3 dimensions, got {data.ndim}")

            data = data.copy()  # Avoid modifying the original data
            data[~mask[..., 0]] = 0
            data_nan = data.astype(np.float64).copy()
            data_nan[~mask[..., 0]] = np.nan
            data_min = data.astype(np.float64).copy()
            data_min[~mask[..., 0]] = float("inf")
            data_max = data.astype(np.float64).copy()
            data_max[~mask[..., 0]] = float("-inf")
            if key not in self.counts:
                sum_mask = np.sum(mask, axis=0)
                sum_data = np.sum(data, axis=0)
                mask_sum_mask = sum_mask > 0
                # Use modified sum_mask only for division to avoid division by zero
                sum_mask_for_division = np.where(mask_sum_mask, sum_mask, 1)
                self.running_means[key] = np.where(mask_sum_mask, sum_data / sum_mask_for_division, 0.0)
                # Handle potential NaN/inf values in initial computation
                self.running_means[key] = np.nan_to_num(self.running_means[key], nan=0.0, posinf=0.0, neginf=0.0)

                self.running_m2s[key] = np.where(
                    mask_sum_mask, np.sum((data - self.running_means[key]) ** 2, axis=0), 0.0
                )
                self.running_m2s[key] = np.nan_to_num(self.running_m2s[key], nan=0.0, posinf=0.0, neginf=0.0)
                self.counts[key] = sum_mask
                self.mins[key] = np.where(sum_mask > 0, np.min(data_min, axis=0), float("inf"))
                self.maxs[key] = np.where(sum_mask > 0, np.max(data_max, axis=0), float("-inf"))
                self.samples_for_percentiles[key] = []
                self.sample_mask_for_percentiles[key] = []
            else:
                # Update running statistics
                n_a = self.counts[key]
                mean_a = self.running_means[key]
                m2_a = self.running_m2s[key]

                n_b = np.sum(mask, axis=0)
                # Safely compute mean_b, handling cases where n_b is 0
                nansum_data = np.nansum(data_nan, axis=0)
                # Use np.divide with where parameter to avoid division warnings
                mean_b = np.divide(nansum_data, n_b, out=np.zeros_like(nansum_data), where=n_b > 0)
                # Handle potential NaN/inf values
                mean_b = np.nan_to_num(mean_b, nan=0.0, posinf=0.0, neginf=0.0)

                # Safely compute m2_b
                m2_b = np.nansum((data_nan - mean_b) ** 2, axis=0)
                m2_b = np.nan_to_num(m2_b, nan=0.0, posinf=0.0, neginf=0.0)

                delta = mean_b - mean_a
                total_count = n_a + n_b

                mask_total_count = total_count > 0
                # Use modified total_count only for division to avoid division by zero
                total_count_for_division = np.where(mask_total_count, total_count, 1)
                self.running_means[key] = np.where(
                    mask_total_count, (n_a * mean_a + n_b * mean_b) / total_count_for_division, 0.0
                )

                self.running_m2s[key] = (
                    m2_a + m2_b + np.where(mask_total_count, delta**2 * (n_a * n_b / total_count_for_division), 0.0)
                )

                self.counts[key] = total_count
                self.mins[key] = np.minimum(self.mins[key], np.min(data_min, axis=0))
                self.maxs[key] = np.maximum(self.maxs[key], np.max(data_max, axis=0))

            # Collect samples for percentile computation (with memory limit, using reservoir sampling)
            n_new = np.sum(mask, axis=0)
            if np.all(n_new == 0):
                continue
            if len(self.samples_for_percentiles[key]) < self.max_samples_for_percentiles:
                # Add samples, but subsample if we have too many new samples
                if np.all(
                    np.sum(mask, axis=0) <= (self.max_samples_for_percentiles - len(self.samples_for_percentiles[key]))
                ):
                    self.samples_for_percentiles[key].extend(data.tolist())
                    self.sample_mask_for_percentiles[key].extend(mask.tolist())
                else:
                    # Subsample to fit within limit for each timestep
                    remaining_slots_per_timestep = self.max_samples_for_percentiles - len(
                        self.samples_for_percentiles[key]
                    )
                    # TODO: Jean This is not efficient for large number of timesteps,
                    # we should use a more efficient subsampling method
                    for timestep in range(data.shape[1]):
                        if remaining_slots_per_timestep[timestep] > 0:
                            step = np.sum(mask[:, timestep]) // remaining_slots_per_timestep
                            if step > 0:
                                subsampled = data[:, timestep][::step][:remaining_slots_per_timestep]
                                self.samples_for_percentiles[key].extend(subsampled.tolist())
                                self.sample_mask_for_percentiles[key].extend(
                                    mask[:, timestep][::step][:remaining_slots_per_timestep].tolist()
                                )

            else:  # Buffer full, reservoir sampling
                # Simple reservoir sampling: for each new sample, decide if it replaces an old one
                current_total = len(self.samples_for_percentiles[key])
                for i, new_sample in enumerate(data):
                    # If sample has any valid data
                    # With probability max_samples/(current_total + i + 1), keep this sample
                    if np.any(mask[i]) and np.random.random() < self.max_samples_for_percentiles / (
                        current_total + i + 1
                    ):
                        # Replace a random existing sample
                        replace_idx = np.random.randint(0, self.max_samples_for_percentiles)
                        self.samples_for_percentiles[key][replace_idx] = new_sample.tolist()
                        self.sample_mask_for_percentiles[key][replace_idx] = mask[i].tolist()

    def merge_from_samples(self, samples_batch: List[Dict[str, Any]]):
        """Efficiently merge statistics from a batch of samples."""
        if not self.compute_stats:
            return

        # Process samples in batch for better performance, first concatenate all the samples in a single dict
        sample_lowdim = {}
        for sample in samples_batch:
            for key, data in sample["lowdim"].items():
                if key not in sample_lowdim:
                    sample_lowdim[key] = data[None, ...]
                else:
                    sample_lowdim[key] = np.concatenate([sample_lowdim[key], data[None, ...]], axis=0)
            if "past_mask" in sample and "future_mask" in sample:
                mask = np.logical_or(sample["past_mask"], sample["future_mask"])
                if "mask" not in sample_lowdim:
                    sample_lowdim["mask"] = mask[None, ...]
                else:
                    sample_lowdim["mask"] = np.concatenate([sample_lowdim["mask"], mask[None, ...]], axis=0)
        # Then update the statistics with the concatenated samples
        self.update(sample_lowdim)

    def get_statistics(self) -> Dict[str, Any]:
        """Get final statistics (thread-safe)."""
        if not self.compute_stats:
            return {"statistics_disabled": True}

        stats = {}
        for key in self.counts:
            if any(self.counts[key] > 1):
                # Safe division for variance calculation
                count_minus_one = self.counts[key] - 1
                mask_count_minus_one = count_minus_one > 0
                count_minus_one = np.where(mask_count_minus_one, count_minus_one, 1)
                variance = np.where(mask_count_minus_one, self.running_m2s[key] / count_minus_one, 0.0)
                # Handle potential NaN/inf values in variance
                variance = np.nan_to_num(variance, nan=0.0, posinf=0.0, neginf=0.0)
                std = np.sqrt(np.maximum(variance, 0.0)).tolist()  # Ensure non-negative before sqrt
            else:
                std = 0.0

            # Compute percentiles if we have enough samples
            percentile_5, percentile_95 = None, None
            if key in self.samples_for_percentiles and len(self.samples_for_percentiles[key]) > 0:
                samples_array = np.array(self.samples_for_percentiles[key])
                samples_array[~np.array(self.sample_mask_for_percentiles[key])[..., 0]] = np.nan
                percentiles = np.nanpercentile(
                    samples_array.reshape(-1, samples_array.shape[-1]), [1, 5, 95, 99], axis=0
                )
                percentile_1, percentile_5, percentile_95, percentile_99 = percentiles.tolist()
                per_timestep_percentiles = np.nanpercentile(samples_array, [1, 5, 95, 99], axis=0)
                (
                    percentile_1_per_timestep,
                    percentile_5_per_timestep,
                    percentile_95_per_timestep,
                    percentile_99_per_timestep,
                ) = per_timestep_percentiles.tolist()

            stats[key] = {
                "mean": np.mean(self.running_means[key], axis=0).tolist(),
                "std": np.std(self.running_means[key], axis=0).tolist(),
                "min": np.min(self.mins[key], axis=0).tolist(),
                "max": np.max(self.maxs[key], axis=0).tolist(),
                "mean_per_timestep": self.running_means[key].tolist(),
                "std_per_timestep": std,
                "min_per_timestep": self.mins[key].tolist(),
                "max_per_timestep": self.maxs[key].tolist(),
                "percentile_5": percentile_5,
                "percentile_95": percentile_95,
                "percentile_1": percentile_1,
                "percentile_99": percentile_99,
                "percentile_1_per_timestep": percentile_1_per_timestep,
                "percentile_5_per_timestep": percentile_5_per_timestep,
                "percentile_95_per_timestep": percentile_95_per_timestep,
                "percentile_99_per_timestep": percentile_99_per_timestep,
                "count": self.counts[key][..., 0].tolist(),
                "percentile_sample_count": np.sum(self.sample_mask_for_percentiles[key], axis=0)[..., 0].tolist(),
            }

        return stats

    def save_state(self, filepath: str):
        """Save the current state of the statistics computation for recovery."""
        if not self.compute_stats:
            return

        # Convert numpy arrays to lists for JSON serialization
        state = {
            "compute_stats": self.compute_stats,
            "max_samples_for_percentiles": self.max_samples_for_percentiles,
            "running_means": {k: v.tolist() for k, v in self.running_means.items()},
            "running_m2s": {k: v.tolist() for k, v in self.running_m2s.items()},
            "counts": {k: v.tolist() for k, v in self.counts.items()},
            "mins": {k: v.tolist() for k, v in self.mins.items()},
            "maxs": {k: v.tolist() for k, v in self.maxs.items()},
            "samples_for_percentiles": self.samples_for_percentiles,
            "sample_mask_for_percentiles": self.sample_mask_for_percentiles,
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load a previously saved state to resume statistics computation."""
        if not self.compute_stats:
            return

        try:
            with open(filepath, "r") as f:
                state = json.load(f)

            # Restore configuration
            self.max_samples_for_percentiles = state.get("max_samples_for_percentiles", 100000)

            # Convert lists back to numpy arrays
            self.running_means = {k: np.array(v) for k, v in state.get("running_means", {}).items()}
            self.running_m2s = {k: np.array(v) for k, v in state.get("running_m2s", {}).items()}
            self.counts = {k: np.array(v) for k, v in state.get("counts", {}).items()}
            self.mins = {k: np.array(v) for k, v in state.get("mins", {}).items()}
            self.maxs = {k: np.array(v) for k, v in state.get("maxs", {}).items()}
            self.samples_for_percentiles = state.get("samples_for_percentiles", {})
            self.sample_mask_for_percentiles = state.get("sample_mask_for_percentiles", {})

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load statistics state from {filepath}: {e}")
            # Reset to empty state
            self.running_means = {}
            self.running_m2s = {}
            self.counts = {}
            self.mins = {}
            self.maxs = {}
            self.samples_for_percentiles = {}
            self.sample_mask_for_percentiles = {}

    @classmethod
    def from_saved_state(cls, filepath: str, compute_stats: bool = True, max_samples_for_percentiles: int = 100000):
        """Create a new instance from a saved state file."""
        instance = cls(compute_stats=compute_stats, max_samples_for_percentiles=max_samples_for_percentiles)
        if compute_stats:
            instance.load_state(filepath)
        return instance
