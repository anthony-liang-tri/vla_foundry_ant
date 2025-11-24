"""Tests for point cloud generation from depth and RGB images."""

import numpy as np
import pytest

from vla_foundry.data.preprocessing.utils import depth_images_to_point_cloud


class TestDepthImagesToPointCloud:
    """Test the depth_images_to_point_cloud function."""

    @pytest.fixture
    def simple_camera_setup(self):
        """Create a simple single-camera setup with depth, RGB, intrinsics, and extrinsics."""
        # Create synthetic depth image (480x640, values in mm)
        depth_img = np.random.randint(1000, 3000, (480, 640), dtype=np.uint16)

        # Create synthetic RGB image
        rgb_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Standard camera intrinsics (fx, fy, cx, cy)
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])

        # Identity extrinsics (camera at world origin)
        Rt = np.eye(4, dtype=np.float32)

        return {
            "depth_images": {"camera1": depth_img},
            "rgb_images": {"camera1": rgb_img},
            "intrinsics": {"camera1": K},
            "extrinsics": {"camera1": Rt},
        }

    @pytest.fixture
    def multi_camera_setup(self):
        """Create a multi-camera setup with 3 cameras."""
        depth_images = {}
        rgb_images = {}
        intrinsics = {}
        extrinsics = {}

        for i, camera_name in enumerate(["left", "center", "right"]):
            # Depth and RGB images
            depth_images[camera_name] = np.random.randint(1000, 3000, (480, 640), dtype=np.uint16)
            rgb_images[camera_name] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # Intrinsics (same for all cameras)
            intrinsics[camera_name] = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])

            # Extrinsics (cameras at different positions)
            Rt = np.eye(4, dtype=np.float32)
            Rt[0, 3] = (i - 1) * 0.1  # Shift along x-axis: -0.1, 0, 0.1
            extrinsics[camera_name] = Rt

        return {
            "depth_images": depth_images,
            "rgb_images": rgb_images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
        }

    def test_output_shape_and_dtype(self, simple_camera_setup):
        """Test that output has correct shape (N, 6) and dtype (float16)."""
        num_points = 1000
        point_cloud = depth_images_to_point_cloud(
            **simple_camera_setup,
            num_points=num_points,
        )

        # Verify shape: (N, 6) where 6 = [x, y, z, r, g, b]
        assert point_cloud.shape == (num_points, 6)

        # Verify dtype
        assert point_cloud.dtype == np.float16

    def test_rgb_colors_in_valid_range(self, simple_camera_setup):
        """Test that RGB colors are in [0, 1] range."""
        num_points = 1000
        point_cloud = depth_images_to_point_cloud(
            **simple_camera_setup,
            num_points=num_points,
        )

        # Extract RGB values (last 3 columns)
        rgb_values = point_cloud[:, 3:6]

        # Verify RGB colors are in [0, 1] range
        assert np.all(rgb_values >= 0.0)
        assert np.all(rgb_values <= 1.0)

    def test_xyz_coordinates_non_zero(self, simple_camera_setup):
        """Test that XYZ coordinates are not all zeros (regression test for bug)."""
        num_points = 1000
        point_cloud = depth_images_to_point_cloud(
            **simple_camera_setup,
            num_points=num_points,
        )

        # Extract XYZ values (first 3 columns)
        xyz_values = point_cloud[:, :3]

        # Verify that at least some XYZ coordinates are non-zero
        assert not np.all(xyz_values == 0.0)

        # Verify that XYZ values are reasonable (not all identical)
        assert np.std(xyz_values) > 0.0

    def test_z_coordinates_positive(self, simple_camera_setup):
        """Test that Z coordinates are positive (points above ground plane)."""
        num_points = 1000
        point_cloud = depth_images_to_point_cloud(
            **simple_camera_setup,
            num_points=num_points,
        )

        # Extract Z values (third column)
        z_values = point_cloud[:, 2]

        # Verify all Z values are >= 0 (ground plane filtering)
        assert np.all(z_values >= 0.0)

    def test_multi_camera_fusion(self, multi_camera_setup):
        """Test that multi-camera point clouds are fused correctly."""
        num_points = 5000
        point_cloud = depth_images_to_point_cloud(
            **multi_camera_setup,
            num_points=num_points,
        )

        # Verify output shape
        assert point_cloud.shape == (num_points, 6)

        # Verify non-zero points (fusion should work)
        xyz_values = point_cloud[:, :3]
        assert not np.all(xyz_values == 0.0)

        # Verify RGB colors are valid
        rgb_values = point_cloud[:, 3:6]
        assert np.all(rgb_values >= 0.0) and np.all(rgb_values <= 1.0)

    def test_invalid_depth_filtering(self):
        """Test that function generates point cloud even with mixed depth values."""
        # Create depth image with some invalid values
        depth_img = np.zeros((100, 100), dtype=np.uint16)
        depth_img[20:80, 20:80] = 2000  # Valid depth in center
        depth_img[0:10, :] = 0  # Zero depth
        depth_img[90:100, :] = 10000  # Large depth

        rgb_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        K = np.array([[500.0, 0.0, 50.0], [0.0, 500.0, 50.0], [0.0, 0.0, 1.0]])
        Rt = np.eye(4, dtype=np.float32)

        point_cloud = depth_images_to_point_cloud(
            depth_images={"cam": depth_img},
            rgb_images={"cam": rgb_img},
            intrinsics={"cam": K},
            extrinsics={"cam": Rt},
            num_points=100,
        )

        # Should successfully generate point cloud
        assert point_cloud.shape == (100, 6)
        assert not np.all(point_cloud[:, :3] == 0.0)

    def test_downsampling_to_target_points(self, simple_camera_setup):
        """Test that output is downsampled to exactly num_points."""
        target_points = 1234
        point_cloud = depth_images_to_point_cloud(
            **simple_camera_setup,
            num_points=target_points,
        )

        # Verify exact number of points
        assert len(point_cloud) == target_points

    def test_voxel_size_parameter(self, simple_camera_setup):
        """Test that voxel_size parameter affects downsampling."""
        num_points = 1000

        # Generate with different voxel sizes
        pc_small_voxel = depth_images_to_point_cloud(
            **simple_camera_setup,
            num_points=num_points,
            voxel_size=0.001,  # 1mm voxels
        )

        pc_large_voxel = depth_images_to_point_cloud(
            **simple_camera_setup,
            num_points=num_points,
            voxel_size=0.01,  # 10mm voxels
        )

        # Both should have target number of points after random downsampling
        assert pc_small_voxel.shape == (num_points, 6)
        assert pc_large_voxel.shape == (num_points, 6)

    def test_empty_input_returns_none(self):
        """Test that requesting more points than available after voxeling returns None."""
        # Create small depth image that won't have enough points after voxeling
        depth_img = np.full((10, 10), 2000, dtype=np.uint16)  # Small 10x10 image
        rgb_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        K = np.array([[500.0, 0.0, 5.0], [0.0, 500.0, 5.0], [0.0, 0.0, 1.0]])
        Rt = np.eye(4, dtype=np.float32)

        num_points = 100000  # Request way more points than possible
        point_cloud = depth_images_to_point_cloud(
            depth_images={"cam": depth_img},
            rgb_images={"cam": rgb_img},
            intrinsics={"cam": K},
            extrinsics={"cam": Rt},
            num_points=num_points,
        )

        # Should return None when not enough points
        assert point_cloud is None

    def test_depth_to_meters_conversion(self):
        """Test that depth values are correctly converted from mm to meters."""
        # Create depth image with known values
        depth_img = np.full((100, 100), 2000, dtype=np.uint16)  # 2000mm = 2m
        rgb_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Simple intrinsics (fx=fy=1, cx=cy=50)
        K = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 50.0], [0.0, 0.0, 1.0]])

        # Identity extrinsics
        Rt = np.eye(4, dtype=np.float32)

        point_cloud = depth_images_to_point_cloud(
            depth_images={"cam": depth_img},
            rgb_images={"cam": rgb_img},
            intrinsics={"cam": K},
            extrinsics={"cam": Rt},
            num_points=100,
        )

        # Check that Z coordinates are approximately 2 meters (with some tolerance for random sampling)
        z_values = point_cloud[:, 2]
        assert np.all(z_values > 0.0)
        # Most points should be close to 2m (since all depth is 2000mm)
        assert np.median(z_values) > 1.5  # Allow some variance from downsampling

    def test_extrinsics_transformation(self):
        """Test that extrinsics correctly transform points to world coordinates."""
        depth_img = np.full((100, 100), 1000, dtype=np.uint16)  # 1m depth
        rgb_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        K = np.array([[500.0, 0.0, 50.0], [0.0, 500.0, 50.0], [0.0, 0.0, 1.0]])

        # Extrinsics with translation
        Rt = np.eye(4, dtype=np.float32)
        Rt[0, 3] = 1.0  # Translate 1m along x-axis
        Rt[1, 3] = 0.5  # Translate 0.5m along y-axis

        point_cloud = depth_images_to_point_cloud(
            depth_images={"cam": depth_img},
            rgb_images={"cam": rgb_img},
            intrinsics={"cam": K},
            extrinsics={"cam": Rt},
            num_points=100,
        )

        # Points should be shifted by translation
        x_values = point_cloud[:, 0]
        y_values = point_cloud[:, 1]

        # Mean should be close to translation values (with some variance)
        assert np.mean(x_values) > 0.5  # Should be around 1.0
        assert np.mean(y_values) > 0.0  # Should be around 0.5


class TestPointCloudEdgeCases:
    """Test edge cases for point cloud generation."""

    def test_single_valid_pixel(self):
        """Test with mostly zero depth values."""
        depth_img = np.zeros((10, 10), dtype=np.uint16)  # Small image
        depth_img[5, 5] = 2000  # One non-zero pixel

        rgb_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        K = np.array([[500.0, 0.0, 5.0], [0.0, 500.0, 5.0], [0.0, 0.0, 1.0]])
        Rt = np.eye(4, dtype=np.float32)

        # Request more points than available after voxeling
        num_points = 100
        point_cloud = depth_images_to_point_cloud(
            depth_images={"cam": depth_img},
            rgb_images={"cam": rgb_img},
            intrinsics={"cam": K},
            extrinsics={"cam": Rt},
            num_points=num_points,
        )

        # Should return None (not enough points after voxeling)
        assert point_cloud is None

    def test_very_small_num_points(self):
        """Test with very small number of requested points."""
        depth_img = np.random.randint(1000, 3000, (100, 100), dtype=np.uint16)
        rgb_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        K = np.array([[500.0, 0.0, 50.0], [0.0, 500.0, 50.0], [0.0, 0.0, 1.0]])
        Rt = np.eye(4, dtype=np.float32)

        num_points = 10  # Very small
        point_cloud = depth_images_to_point_cloud(
            depth_images={"cam": depth_img},
            rgb_images={"cam": rgb_img},
            intrinsics={"cam": K},
            extrinsics={"cam": Rt},
            num_points=num_points,
        )

        assert point_cloud.shape == (num_points, 6)

    def test_large_num_points(self):
        """Test with large number of requested points."""
        depth_img = np.random.randint(1000, 3000, (480, 640), dtype=np.uint16)
        rgb_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        Rt = np.eye(4, dtype=np.float32)

        num_points = 100000  # Very large
        point_cloud = depth_images_to_point_cloud(
            depth_images={"cam": depth_img},
            rgb_images={"cam": rgb_img},
            intrinsics={"cam": K},
            extrinsics={"cam": Rt},
            num_points=num_points,
        )

        assert point_cloud.shape == (num_points, 6)

    def test_mismatched_camera_names(self):
        """Test that mismatched camera names raise appropriate errors."""
        depth_img = np.random.randint(1000, 3000, (100, 100), dtype=np.uint16)
        rgb_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        K = np.array([[500.0, 0.0, 50.0], [0.0, 500.0, 50.0], [0.0, 0.0, 1.0]])
        Rt = np.eye(4, dtype=np.float32)

        # Mismatched keys
        with pytest.raises(KeyError):
            depth_images_to_point_cloud(
                depth_images={"cam1": depth_img},
                rgb_images={"cam2": rgb_img},  # Different key
                intrinsics={"cam1": K},
                extrinsics={"cam1": Rt},
                num_points=100,
            )


class TestPointCloudRealisticScenarios:
    """Test realistic robotics scenarios."""

    def test_robot_wrist_cameras(self):
        """Test with typical robot wrist camera configuration (4 cameras)."""
        camera_names = ["wrist_left", "wrist_right", "scene_left", "scene_right"]
        depth_images = {}
        rgb_images = {}
        intrinsics = {}
        extrinsics = {}

        for i, camera_name in enumerate(camera_names):
            depth_images[camera_name] = np.random.randint(500, 3000, (480, 640), dtype=np.uint16)
            rgb_images[camera_name] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # Slightly different intrinsics per camera
            intrinsics[camera_name] = np.array(
                [[500.0 + i * 10, 0.0, 320.0], [0.0, 500.0 + i * 10, 240.0], [0.0, 0.0, 1.0]]
            )

            # Different poses for each camera
            Rt = np.eye(4, dtype=np.float32)
            angle = i * np.pi / 2  # 0, 90, 180, 270 degrees
            Rt[0, 3] = np.cos(angle) * 0.2
            Rt[1, 3] = np.sin(angle) * 0.2
            extrinsics[camera_name] = Rt

        # Typical robot point cloud size
        num_points = 50000
        point_cloud = depth_images_to_point_cloud(
            depth_images=depth_images,
            rgb_images=rgb_images,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            num_points=num_points,
        )

        # Verify output
        assert point_cloud.shape == (num_points, 6)
        assert not np.all(point_cloud[:, :3] == 0.0)
        assert np.all(point_cloud[:, 3:6] >= 0.0) and np.all(point_cloud[:, 3:6] <= 1.0)

    def test_temporal_sequence(self):
        """Test generating point clouds for a temporal sequence (multiple timesteps)."""
        num_timesteps = 5
        num_points = 1000

        point_clouds = []
        for t in range(num_timesteps):
            # Simulate robot moving (depth changes over time)
            depth_img = np.random.randint(1000 + t * 100, 2000 + t * 100, (100, 100), dtype=np.uint16)
            rgb_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            K = np.array([[500.0, 0.0, 50.0], [0.0, 500.0, 50.0], [0.0, 0.0, 1.0]])
            Rt = np.eye(4, dtype=np.float32)

            pc = depth_images_to_point_cloud(
                depth_images={"cam": depth_img},
                rgb_images={"cam": rgb_img},
                intrinsics={"cam": K},
                extrinsics={"cam": Rt},
                num_points=num_points,
            )
            point_clouds.append(pc)

        # Stack into (T, N, 6) array (like in spartan.py)
        point_clouds_array = np.stack(point_clouds, axis=0)

        assert point_clouds_array.shape == (num_timesteps, num_points, 6)

        # Verify temporal consistency (all timesteps have valid data)
        for t in range(num_timesteps):
            assert not np.all(point_clouds_array[t, :, :3] == 0.0)
