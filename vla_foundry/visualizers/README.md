# VLA Foundry Visualizers

This directory contains the visualization tools for the VLA Foundry project. These tools allow you to log and visualize data such as images, 3D points, trajectories, and scalar values during training or debugging. The visualizers are designed to be modular and support multiple backends.

## Available Visualizers

### 1. **Rerun Visualizer**
   - **File**: `rerun_backend.py`
   - **Purpose**: Logs data to the [Rerun.io](https://rerun.io/) visualization platform.
   - **Use Cases**:
     - Visualizing 3D points, trajectories, and rigid transforms.
     - Logging images and scalar values.
     - Debugging robot arm poses and action predictions.
   - **Key Features**:
     - Supports hierarchical logging paths.
     - Provides decorators for logging images and robot arm poses.

### 2. **Gradio Visualizer**
   - **File**: `gradio_backend.py`
   - **Purpose**: Provides a lightweight, interactive web-based visualization using [Gradio](https://gradio.app/).
   - **Use Cases**:
     - Visualizing images, scalars, and 3D data in a browser.
     - Creating interactive dashboards for debugging and monitoring.
   - **Key Features**:
     - Displays images, scalars, and 3D plots dynamically.
     - Includes sliders for navigating through logged data.

### 3. **Rerun Visualizer Utilities**
   - **File**: `rerun_visualizers.py`
   - **Purpose**: Provides utility functions and decorators for logging data to Rerun.io.
   - **Use Cases**:
     - Simplifying the process of logging images and robot arm poses.
     - Adding visualization capabilities to existing functions using decorators.

### 4. **Visualizer Facade**
   - **File**: `visualizer.py`
   - **Purpose**: Provides a unified interface for all visualizers.
   - **Use Cases**:
     - Abstracting backend-specific details.
     - Automatically selecting the appropriate backend based on the `VISUALIZER` environment variable.
   - **Key Features**:
     - Supports multiple backends (`rerun`, `gradio`, or `disabled`).
     - Automatically disables visualization if no backend is selected.

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
  - `gradio`: Use the Gradio backend.
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

### Gradio Backend
- **Best For**: Lightweight, interactive web-based visualization.
- **Setup**: Install the `gradio` Python package.
- **Environment Variable**: `VISUALIZER=gradio`

### Disabled
- **Best For**: Running without visualization (e.g., in production or testing environments).
- **Setup**: No additional setup required.
- **Environment Variable**: `VISUALIZER=disabled`

---

## Example Usage

### Logging Images and Scalars
```python
import vla_foundry.visualizers as vz
import numpy as np

# Initialize the visualizer
vz.init(run_name="example_logging", add_rank_to_run=True)

# Log an image
image = np.random.rand(100, 100, 3)
vz.log_image("example/image", image)

# Log a scalar
vz.log_scalar("example/scalar", 3.14)

# Shutdown the visualizer
vz.shutdown()
```

### Logging 3D Data
```python
import vla_foundry.visualizers as vz
import numpy as np

# Initialize the visualizer
vz.init(run_name="example_3d_logging")

# Log 3D points
points = np.random.rand(10, 3)
vz.log_points3d("example/3d_points", points)

# Log a trajectory
trajectory = np.random.rand(20, 3)
vz.log_trajectory("example/trajectory", trajectory)

# Shutdown the visualizer
vz.shutdown()
```

---

## Notes
- The `visualizer.py` facade automatically disables visualization if no backend is selected.
- Use the `VISUALIZER` environment variable to control the backend selection.
- For more advanced use cases, refer to the backend-specific files (`rerun_backend.py`, `gradio_backend.py`, etc.).
