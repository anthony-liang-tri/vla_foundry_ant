import numpy as np


def process_robotics_sample(sample, action_token_text=None, image_token_text=None):
    """Process robotics sample into standardized format."""
    processed = {}

    # Extract images
    images = {}
    lowdim_data = None
    masks_data = None
    metadata = None
    actions_data = None
    intrinsics = None
    extrinsics = None

    for key, value in sample.items():
        if key.endswith(".jpg"):
            # Extract camera name and timestep from key (format: {sample_id}.{camera}_{timestep}.jpg)
            img_key = key.split(".")[-2]  # e.g., "wrist_camera_t-1"
            images[img_key] = np.array(value)
        # Extract lowdim data
        elif key.endswith("lowdim.npz"):
            lowdim_data = value
        # Extract masks
        elif key.endswith("masks.npz"):
            masks_data = value
        # Extract metadata
        elif key.endswith("metadata.json"):
            metadata = value
        # Extract actions (if present)
        elif key.endswith("actions.npz"):
            actions_data = value
        # Extract camera calibration data
        elif key.endswith("intrinsics.npz"):
            intrinsics = value
        elif key.endswith("extrinsics.npz"):
            extrinsics = value

    if image_token_text is not None:
        num_images = len(images)
        # TODO: Jean Here we assume that language instructions are provided for each time step but that they are all
        # the same and so we take the first one. We might want to change this to handle changing instructions.
        lowdim_data["language_instruction"] = [
            image_token_text * num_images + " " + lowdim_data["language_instruction"][0]
        ]
    if action_token_text is not None:
        lowdim_data["language_instruction"] = [v + " " + action_token_text for v in lowdim_data["language_instruction"]]

    # Structure the processed sample
    processed = {
        "images": images,
        "lowdim": lowdim_data if lowdim_data is not None else {},
        "masks": masks_data if masks_data is not None else {},
        "actions": actions_data if actions_data is not None else {},
        "metadata": metadata if metadata is not None else {},
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
    }

    return processed
