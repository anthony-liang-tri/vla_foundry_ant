.venv/bin/torchrun --nproc_per_node=8 --nnodes=1 lbm2/main.py \
--model.type transformer \
--model "include lbm2/config_presets/models/transformer_410m.yaml" \
--distributed.fsdp True \
--data.type text_untokenized \
--data.dataset_manifest ["s3://tri-ml-datasets/slbm2_datasets/text/fineweb-edu-350BT/manifest.jsonl"] \
--data.dataset_modality ["text_untokenized"] \
--data.dataset_weighting [1.0] \
--data.tokenizer HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
--data.seq_len 4096 \
--data.allow_multiple_epochs True \
--total_train_samples 40_000_000 \
--num_checkpoints 50 \
--hparams.per_gpu_batch_size 2 \
--hparams.global_batch_size 512 \
--remote_sync s3://tri-ml-datasets/lbm2_models/llm_410m_fineweb_edu_350BT_cc20

# 410M * 20 * cc20 / 4096 = 40_000_000
