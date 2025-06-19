(todo)
- gradient accumulation
- fsdp, torchrun, distributed
- what if train samples is more/less than dataset size
- seeds, reproducibility


todos / thought dump:
- check optimizer scaler
[x] load multiple datasets
- load dataset from path instead of manifest
[x] indicate training based on num epochs
[x] mixing datasets
[x] mixing modalities
[x] shard shuffle seed
[x] multiple epochs over the dataset
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
[x] check why the number of checkpoints isn't ways the same as what's indicated
[x] check that inference didn't break while making changes + check vlm inference
- double check webdataset speed if it's faster to conver to tensor inside the pipeline or return lists then convert outside
- double check fp 16 bf 16
- precompute masks if static
- text intermediate layer as parameter
- maybe support processor_configs instead of merging into data/model/vit?
[x] load siglip weights
[x] load paligemma weights
- user-specified processor instead of loading from existing
[x] log which tar files are used in which checkpoints
- support processors beyond just pali gemma
[x] vlm inference
- action pipeline
- diffusion policy simple implementation
- kv cache for inference
[x] allow for freezing parts of the VLM
- prefix lm
- double check if siglip loading step is correct (specifically intermediate layers)
- make attention implementation cleaner / more unified
[x] loss selector
- support distributed mode without FSDP
- fix naming
- text-conditioned stable diffusion

Known bugs / improvements
- speed up pretokenized huggingface? 
- torchcompile for accum_freq > 1
- speed up vlm mask computation inside forward loop

Ideas that probably won't be implemented
- sampling with replacement --> Our sampling without replacement is now working robustly across different scenarios so this seems unnecessary.
