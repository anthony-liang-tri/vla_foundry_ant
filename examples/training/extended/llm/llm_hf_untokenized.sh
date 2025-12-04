.venv/bin/torchrun --nproc_per_node=8 --nnodes=1 vla_foundry/main.py \
--model.type transformer_hf \
--model.hf_pretrained Qwen/Qwen2.5-0.5B \
--distributed.fsdp True \
--data.type text_untokenized \
--data.dataset_manifest ["s3://tri-ml-datasets/vla_foundry_scratch/datasets/synthetic-untokenized/manifest.jsonl"] \
--data.dataset_modality ["text_untokenized"] \
--data.dataset_weighting [1.0] \
--data.tokenizer Qwen/Qwen2.5-0.5B \
--data.seq_len 2048 \
--data.allow_multiple_epochs True \
--total_train_samples 14_000_000 \
--num_checkpoints 5 \
--hparams.per_gpu_batch_size 8 \
--hparams.global_batch_size 512 \
--remote_sync s3://tri-ml-datasets/vla_foundry_scratch/models/llm_hf_untokenized \
"$@"
