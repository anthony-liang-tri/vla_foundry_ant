export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
.venv/bin/torchrun --nproc_per_node=8 --nnodes=1 lbm2/main.py \
--model "include lbm2/config_presets/models/vlm_3b.yaml" \
--distributed.fsdp True \
--data.type image_caption \
--data.processor google/paligemma-3b-pt-224 \
--data.dataset_manifest ["s3://tri-ml-datasets/datasets/datacompdr_1b/manifest.jsonl"] \
--data.dataset_modality ["image_caption"] \
--data.dataset_weighting [1.0] \
--data.seq_len 2048 \
--data.img_num_tokens 256 \
--total_train_samples 14_000_000 \
--num_checkpoints 5 \
--hparams.per_gpu_batch_size 2 \
--hparams.global_batch_size 128 \
--remote_sync s3://tri-ml-datasets/scratch/sedrick.keh/sedrick/vlm_paligemma_3b