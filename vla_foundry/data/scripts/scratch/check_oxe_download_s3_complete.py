from vla_foundry.data.preprocessing.hf_utils.hf_downloader_utils import check_lerobot_complete

datasets = [
    "austin_sailor_dataset_lerobot",
    "austin_sirius_dataset_lerobot",
    "bc_z_lerobot",
    "berkeley_autolab_ur5_lerobot",
    "berkeley_cable_routing_lerobot",
    "berkeley_fanuc_manipulation_lerobot",
    "berkeley_mvp_lerobot",
    "berkeley_rpt_lerobot",
    "bridge_orig_lerobot",
    "cmu_play_fusion_lerobot",
    "cmu_stretch_lerobot",
    "dlr_edan_shared_control_lerobot",
    "dobbe_lerobot",
    "droid_lerobot",
    "fmb_dataset_lerobot",
    "fractal20220817_data_lerobot",
    "furniture_bench_dataset_lerobot",
    "iamlab_cmu_pickup_insert_lerobot",
    "jaco_play_lerobot",
    "kuka_lerobot",
    "language_table_lerobot",
    "nyu_door_opening_surprising_effectiveness_lerobot",
    "nyu_franka_play_dataset_lerobot",
    "roboturk_lerobot",
    "stanford_hydra_dataset_lerobot",
    "taco_play_lerobot",
    "toto_lerobot",
    "ucsd_kitchen_dataset_lerobot",
    "utaustin_mutex_lerobot",
    "viola_lerobot",
]

for dataset in datasets:
    missing_files = check_lerobot_complete(f"s3://tri-ml-datasets/hf_datasets/oxe_lerobot/{dataset}/")
    print(f"{dataset}: {len(missing_files)} missing files")
    print(missing_files)
