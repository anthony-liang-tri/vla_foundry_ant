# lbm2

This was put together from some combination of [MBM](https://github.com/mlfoundations/mbm) and [nanoVLM](https://github.com/huggingface/nanoVLM/tree/main).

## Quickstart
.

### Running on SageMaker
To install SageMaker,
```
pip install install/sagemaker-2.240.1.dev0.tar
```

## Technical Details
.

### Param/Argument Structure
.

### Dataloader
.

### Tests
(todo)
- gradient accumulation
- fsdp, torchrun, distributed
- what if train samples is more/less than dataset size
- seeds, reproducibility


todos:
- check optimizer scaler
[x] load multiple datasets
- load dataset from path instead of manifest
- load dataset with sampling
- indicate training based on num epochs
- sampling with replacement
[x] mixing datasets
[x] mixing modalities
- shard shuffle seed
- hf wrapper

- double check reproducibility / randomness
--- check as well when restarting from checkpoints that data loading randomness is the same
[x] saving checkpoints locally
--- make sure to save not just the weights but also the dataloader / optimizer states
[x] saving checkpoints to remote
[x] creating experiment logs folder and saving configs
[x] loading checkpoints
-- resuming from run with optimizer states
-- just loading checkpoints but not resuming from run
[x] generation / inference
[x] fix wandb logging
[x] allow warmup by percentage
[x] from_pretrained huggingface
[x] tokenize on the fly
[x] correct masking for tokenizing on the fly
[x] double check tok/sec/gpu vs seq/sec/gpu make sure logging correctly
[x] handle batch size vs accum freq

- tests to be able to test out each component individually
- tokenized dataset creation script
[x] sagemaker launcher
- check why the number of checkpoints isn't ways the same as what's indicated
- check that inference didn't break while making changes + check vlm inference

Known bugs / improvements
- speed up pretokenized huggingface? 
- torchcompile for accum_freq > 1