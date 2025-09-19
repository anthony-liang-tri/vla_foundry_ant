import numpy as np
import pytest

from lbm2.data.scripts.preprocessing.preprocess_statistics import StreamingDatasetStatistics


@pytest.mark.parametrize(
    "mean,std,num_samples,shape",
    [
        (-3.0, 0.5, 5000, (1, 3, 2)),
        (0.0, 1.0, 1000, (4, 10, 3)),
        (5.0, 2.0, 500, (2, 5, 6)),
    ],
)
def test_streaming_dataset_statistics_basic(mean, std, num_samples, shape):
    """
    Test the StreamingDatasetStatistics class with a simple normal distribution.
    The shape is (batch_size, num_timesteps, num_channels)
    The test gives one batch of samples at a time, and the StreamingDatasetStatistics class is updated with each batch.
    The test checks that the statistics are computed correctly by comparing them to the global statistics computed on
    all the samples.
    """
    np.random.seed(42)
    max_samples_for_percentiles = 1000
    stats = StreamingDatasetStatistics(compute_stats=True, max_samples_for_percentiles=max_samples_for_percentiles)
    key = "test_key"
    all_data = []
    all_mask = []
    for _ in range(0, num_samples, shape[0]):
        data = np.random.normal(loc=mean, scale=std, size=shape).astype(np.float32)
        mask = np.random.randint(0, 2, (shape[0], shape[1]), dtype=bool)
        sample = {key: data.copy(), "mask": mask.copy()}
        all_data.append(data)
        all_mask.append(mask)
        stats.update(sample)
    all_data = np.concatenate(all_data, axis=0)
    all_mask = np.concatenate(all_mask, axis=0)
    all_data[~all_mask] = 0
    expected_mean = np.sum(all_data, axis=0) / np.sum(all_mask, axis=0)[..., None]
    squared_diff = (all_data - expected_mean) ** 2
    squared_diff[~all_mask] = 0
    expected_std = np.sqrt(np.sum(squared_diff, axis=0) / np.sum(all_mask, axis=0)[..., None])
    all_data_no_min = all_data.copy()
    all_data_no_min[~all_mask] = float("inf")
    expected_min = np.min(all_data_no_min, axis=0)
    all_data_no_max = all_data.copy()
    all_data_no_max[~all_mask] = float("-inf")
    expected_max = np.max(all_data_no_max, axis=0)

    expected_percentiles = np.percentile(all_data[all_mask].reshape(-1, shape[2]), [1, 5, 95, 99], axis=0)
    expected_percentiles_per_timestep = np.zeros((4, shape[1], shape[2]))
    for timestep in range(shape[1]):
        data_timestep = all_data[:, timestep]
        expected_percentiles_per_timestep[:, timestep] = np.percentile(
            data_timestep[all_mask[:, timestep]], [1, 5, 95, 99], axis=0
        )
    computed = stats.get_statistics()[key]

    # Allow small numerical error
    np.testing.assert_allclose(computed["mean_per_timestep"], expected_mean, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(
        computed["std_per_timestep"], expected_std, rtol=1e-1, atol=1e-1
    )  # Tolerance is higher because of the Welford's online algorithm, not exact
    np.testing.assert_allclose(computed["min_per_timestep"], expected_min, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(computed["max_per_timestep"], expected_max, rtol=1e-3, atol=1e-3)
    # Percentiles
    np.testing.assert_allclose(
        computed["percentile_1_per_timestep"], expected_percentiles_per_timestep[0], rtol=1e-1, atol=1e-1
    )
    np.testing.assert_allclose(
        computed["percentile_5_per_timestep"], expected_percentiles_per_timestep[1], rtol=1e-1, atol=1e-1
    )
    np.testing.assert_allclose(
        computed["percentile_95_per_timestep"], expected_percentiles_per_timestep[2], rtol=1e-1, atol=1e-1
    )
    np.testing.assert_allclose(
        computed["percentile_99_per_timestep"], expected_percentiles_per_timestep[3], rtol=1e-1, atol=1e-1
    )
    np.testing.assert_allclose(computed["percentile_1"], expected_percentiles[0], rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(computed["percentile_5"], expected_percentiles[1], rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(computed["percentile_95"], expected_percentiles[2], rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(computed["percentile_99"], expected_percentiles[3], rtol=1e-1, atol=1e-1)

    assert np.array(computed["count"]).sum() == all_mask.sum()
    np.testing.assert_array_compare(np.less_equal, computed["percentile_sample_count"], computed["count"])
