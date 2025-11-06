if __name__ == "__main__":
    lbm_train_tars = {}

    # This file is from
    # s3://robotics-manip-lbm-us-west-2/webdataset/format-v1.2/datasets/lbm-v1.0/lbm/sim/val/indices/filenames.txt
    with open("filenames_val.txt", "r") as f:
        for line in f:
            line = line.strip()
            task = line.removeprefix("s3://robotics-manip-lbm/webdataset/format-v1.2/tarfiles/sim/").split("/")[1]
            episode = line.removeprefix("s3://robotics-manip-lbm/webdataset/format-v1.2/tarfiles/sim/").split("/")[2]
            episode_only = int(episode.split("-")[-1])
            if task in lbm_train_tars:
                lbm_train_tars[task].add(episode_only)
            else:
                lbm_train_tars[task] = set([episode_only])

    print(lbm_train_tars)
    # Result copy-pasted to vla_foundry/data/scripts/scratch/lbm_validation_episodes.json
