# TRI-specific files and READMEs

## Shared Directories
- Clean directory: `s3://tri-ml-datasets-uw2/vla_foundry_datasets/`
    - Please do not add anything here until it has already been tested. 
    - More specifically, it would be nice if every folder/file in that bucket has a corresponding script in this repo that can be ran to reproduce that dataset. 
    - Eventually we probably want to protect this bucket to avoid overwriting. We're small enough now that it's fine. 
- Messy directory for scratch: `s3://tri-ml-datasets-uw2/vla_foundry_scratch/`