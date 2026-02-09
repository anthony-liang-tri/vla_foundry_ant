import subprocess

tasks = {}
tasks_fourcameras = {
    "BimanualHangMugsOnMugHolderFromTable",
    "PutSpatulaInUtensilCrockFromDryingRack",
    "BimanualPutMugsOnPlatesFromDryingRack",
    "BimanualPutSpatulaOnTableFromDryingRack",
    "BimanualPlaceFruitFromBowlOnCuttingBoard",
    "BimanualLayCerealBoxOnCuttingBoardFromUnderShelf",
    "PutSpatulaInUtensilCrock",
    "BimanualPlaceAppleFromBowlOnCuttingBoard",
    "BimanualPlaceAppleFromBowlIntoBin",
    "BimanualPlaceAvocadoFromBowlOnCuttingBoard",
    "BimanualHangMugsOnMugHolderFromDryingRack",
    "BimanualPutMugsOnPlatesFromTable",
    "BimanualPlacePearFromBowlIntoBin",
    "BimanualPutSpatulaOnTableFromUtensilCrock",
    "BimanualPutRedBellPepperInBin",
    "BimanualStackPlatesOnTableFromDryingRack",
    "BimanualLayCerealBoxOnCuttingBoardFromTopShelf",
    "BimanualPutSpatulaOnPlateFromTable",
    "BimanualPlacePearFromBowlOnCuttingBoard",
    "BimanualStackPlatesOnTableFromTable",
    "PickAndPlaceBox",
    "BimanualPutSpatulaOnPlateFromDryingRack",
    "BimanualPlaceFruitFromBowlIntoBin",
    "BimanualStoreCerealBoxUnderShelf",
}
tasks_sixcameras = {
    "PutMugOnSaucer",
    "PutCupOnSaucer",
    "PushCoasterToCenterOfTable",
    "PutGreenAppleOnSaucer",
    "PutOrangeOnSaucer",
    "PutOrangeInCenterOfTable",
    "PutCupInCenterOfTable",
    "TurnMugRightsideUp",
    "PlaceCupOnCoaster",
    "TurnCupUpsideDown",
    "PutBananaInCenterOfTable",
    "PutBananaOnSaucer",
    "PlaceCupByCoaster",
    "PutKiwiInCenterOfTable",
    "PutGreenAppleInCenterOfTable",
    "PushCoasterToMug",
    "PutKiwiOnSaucer",
}
four_camera_names = ["scene_right_0", "scene_left_0", "wrist_left_plus", "wrist_right_minus"]
six_camera_names = [
    "scene_right_0",
    "scene_left_0",
    "wrist_left_minus",
    "wrist_left_plus",
    "wrist_right_minus",
    "wrist_right_plus",
]

with open("vla_foundry/tri/stage3_singletask_sim/stage3_sim_filenames.txt", "r") as f:
    for line in f:
        if not line.strip():
            continue
        task_name = line.strip().removeprefix("s3://robotics-manip-lbm/efs/data/tasks/").split("/")[0]
        if task_name not in tasks:
            tasks[task_name] = []
        tasks[task_name].append(line.strip())


cmd = """
python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--type "spartan" \
--source_episodes {source_episodes} \
--output_dir {output_dir} \
--past_lowdim_steps 1 \
--future_lowdim_steps 14 \
--max_padding_left 1 \
--max_padding_right 7 \
--data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \
--camera_names {camera_names} \
--language_annotations_path vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml \
--action_fields_config_path vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml \
--samples_per_shard 100 \
--config_path vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml \
--validation_episodes_path vla_foundry/data/scripts/scratch/lbm_validation_episodes.json \
--resize_images_size "[342, 256]" \
"""

for task in tasks:
    print("=" * 50, "\n", "Now running task: ", task, "\n", "=" * 50)
    source_episodes = '"[{}]"'.format(", ".join([f"'{episode}'" for episode in tasks[task]]))
    output_dir = f"s3://tri-ml-datasets-uw2/vla_foundry_datasets/stage3_singletask_sim_lbm1replication_train/{task}/"
    if task in tasks_fourcameras:
        camera_names = four_camera_names
    elif task in tasks_sixcameras:
        camera_names = six_camera_names
    else:
        raise ValueError(f"Unknown task: {task}")

    curr_cmd = cmd.format(
        source_episodes=source_episodes, output_dir=output_dir, camera_names=f"[{','.join(camera_names)}]"
    )

    # Run the command
    subprocess.run(curr_cmd, shell=True)
    print("=" * 50, "\n", "Task completed: ", task, "\n", "=" * 50)
