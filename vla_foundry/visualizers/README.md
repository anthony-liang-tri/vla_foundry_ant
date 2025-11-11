# VLA Foundry Visualizers

This directory contains the visualization tools for the VLA Foundry project. These tools allow you to log and visualize data such as images, 3D points, trajectories, and scalar values during training or debugging. The visualizers are designed to be modular and support multiple backends.

## Available Visualizers

### 1. **Visualizer Facade**
   - **File**: `visualizer.py`
   - **Purpose**: Provides a unified interface for all visualizers.
   - **Use Cases**:
     - Abstracting backend-specific details.
     - Automatically selecting the appropriate backend based on the `VISUALIZER` environment variable.
   - **Key Features**:
     - Supports multiple backends (`rerun`, `wandb`, `gradio`, or `disabled`).
     - Automatically disables visualization if no backend is selected. Defaults to `disabled`.

### 2. **Rerun Visualizer**
   - **File**: `rerun_backend.py`
   - **Purpose**: Logs data to the [Rerun.io](https://rerun.io/) visualization platform.
   - **Use Cases**:
     - Visualizing 3D points, trajectories, and rigid transforms.
     - Logging images and scalar values.
     - Debugging robot arm poses and action predictions.
   - **Key Features**:
     - Supports hierarchical logging paths.
     - Provides decorators for logging images and robot arm poses.

### 3. **WandB Visualizer**
   - **File**: `wandb_backend.py`
   - **Purpose**: Logs data to the [Weights & Biases](https://wandb.ai/) experiment tracking platform.
   - **Use Cases**:
     - Tracking scalar metrics, images, 3D points, and trajectories in wandb
     - Useful for integrating into training loops

---

## How to Use the Visualizers

### 1. **Setup**
Ensure you have the required dependencies installed. Follow the environment setup instructions in the main repository README.

### 2. **Initialization**
Use the `visualizer.py` facade to initialize the visualizer. The backend is automatically selected based on the `VISUALIZER` environment variable.

```python
import vla_foundry.visualizers as vz

# Initialize the visualizer
vz.init(run_name="example_run", add_rank_to_run=True)
```

- **Environment Variable**: Set `VISUALIZER` to one of the following:
  - `rerun`: Use the Rerun.io backend.
  - `wandb`: Use the Weights & Biases backend.
  - `disabled`: Disable visualization.

### 3. **Logging Data**
The `visualizer.py` facade provides functions for logging various types of data:

#### Log an Image
```python
import numpy as np

image = np.random.rand(100, 100, 3)  # Example image
vz.log_image("example/image1", image)
```

#### Log Multiple Images
```python
images = {
    "example/image2": np.random.rand(100, 100, 3),
    "example/image3": np.random.rand(100, 100, 3),
}
vz.log_images("example/images", images)
```

#### Log Scalars
```python
vz.log_scalar("example/scalar1", 42.0)
```

#### Log 3D Points
```python
points = np.random.rand(10, 3)  # Example 3D points
vz.log_points3d("example/points", points)
```

#### Log Trajectories
```python
trajectory = np.random.rand(20, 3)  # Example trajectory
vz.log_trajectory("example/trajectory", trajectory)
```

#### Log Rigid Transforms
```python
from pydrake.math import RigidTransform

transform = RigidTransform()  # Example rigid transform
vz.log_rigid_transform("example/transform", transform)
```

### 4. **Shutdown**
The visualizer will automatically shut down at the end of the program. However, you can explicitly call `shutdown` if needed:
```python
vz.shutdown()
```

---

## Choosing a Backend

### Rerun Backend
- **Best For**: Advanced 3D visualization, hierarchical logging, and debugging robot arm poses.
- **Setup**: Install the `rerun` Python package.
- **Environment Variable**: `VISUALIZER=rerun`

### WandB Backend
- **Best For**: Experiment tracking, logging scalar metrics, and visualizing 3D data.
- **Setup**: Install the `wandb` Python package.
- **Environment Variable**: `VISUALIZER=wandb`

### Disabled
- **Best For**: Running without visualization (e.g., in production or testing environments).
- **Setup**: No additional setup required.
- **Environment Variable**: `VISUALIZER=disabled`

## Planned Backends:
- **gradio**: Interactive web-based visualizations.

---

## Example Usage

To see an example of how to use the interface, you can run

```
VISUALIZER=rerun uv run example_usage.py
```
or
```
VISUALIZER=wandb uv run example_usage.py
```

If you run simply
```
uv run example_usage.py
```
visualization will default to `disabled` if no backend is selected.

---

## Notes
- The `visualizer.py` facade automatically disables visualization if no backend is selected.
- Use the `VISUALIZER` environment variable to control the backend selection.
- For more advanced use cases, refer to the backend-specific files (`rerun_backend.py`, `wandb_backend.py`, etc.).
