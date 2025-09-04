# LBM2 Copilot Coding Agent Instructions

## Repository Overview

**LBM2** is a Python codebase for training Large Behavior Models, combining MBM and nanoVLM architectures. The repository supports:
- **Model Types**: Transformers, Vision-Language Models (VLMs), Diffusion models, UNets
- **Size**: ~7000+ lines of Python across 50+ files, medium-large ML research codebase
- **Runtime**: Python 3.10, PyTorch 2.7.0, uses `uv` for dependency management
- **Key Dependencies**: transformers, webdataset, draccus, wandb, s3fs, timm, accelerate

## Environment Setup & Dependencies

**ALWAYS use these exact commands in sequence:**

1. **Install uv package manager first** (if not available):
```bash
pip install uv
```

2. **Setup environment** (takes 3-5 minutes):
```bash
uv sync --frozen  # Use --frozen to avoid lockfile updates
uv pip install -e .
```

3. **Verify installation**:
```bash
uv run python -c "import lbm2; print('LBM2 loaded successfully')"
```

**Environment Notes:**
- Virtual environment created at `.venv/`
- Python 3.10.18 required (auto-installed by uv)
- Uses uv workspace structure with packages in `packages/` (robot-gym, grpc-workspace)
- SageMaker group: Use `uv run --group sagemaker` for SageMaker-related commands
- Two optional dependency groups: `--group sagemaker` and `--group inference`

## Build & Validation Workflow

### 1. Linting (Required before changes)
```bash
# Check linting (takes ~10 seconds)
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```
**Configuration**: pyproject.toml with line-length 120, excludes `packages/`

### 2. Testing (3 categories)
```bash
# Fast local tests (~20 seconds, NO internet required)
uv run pytest tests/models/ tests/params/ -v

# Data pipeline tests (~30 seconds, some may need internet)
uv run pytest tests/data/ -v

# Full test suite (~60 seconds, may have internet failures)
uv run pytest tests/ --verbose
```

**Test Notes:**
- Tests requiring HuggingFace/internet access may fail in sandboxed environments
- Use `tests/shared/tiny_dataset/` for data tests requiring datasets
- Test structure mirrors main code: `tests/data/`, `tests/models/`, etc.

### 3. GitHub Workflows
- **Lint workflow**: Runs `uvx ruff check .`
- **Test workflow**: Runs `uv sync --frozen && uv run pytest --verbose tests/`
- Both require Python 3.10 and use uv

## Key Architecture & File Locations

### Main Entry Points
- **Training**: `lbm2/main.py` - Main training script with draccus argument parsing
- **Examples**: `examples/training/llm_11m.sh` - Good starting point for training examples

### Configuration System (Critical)
- **Base**: `lbm2/params/` - Draccus dataclass-based config with nested structures
- **Presets**: `lbm2/config_presets/models/` - YAML model configurations
- **Pattern**: Use `--model.type transformer --model "include path/to/preset.yaml"`
- **Command-line precedence**: CLI args override YAML presets

### Core Modules
- **Models**: `lbm2/models/` - create_model() factory, transformer/VLM/diffusion implementations
- **Data**: `lbm2/data/` - WebDataset pipelines, processors for different modalities
- **Training**: `lbm2/train.py` - Core training loop, `lbm2/distributed.py` - FSDP/DDP support

### Project Structure
```
lbm2/                 # Main package
├── main.py          # Training entry point
├── params/          # Configuration classes
├── models/          # Model implementations
├── data/            # Data loading & processing
├── config_presets/  # YAML configuration presets
examples/training/   # Training scripts & examples  
tests/               # Test suite (mirrors lbm2/ structure)
packages/            # Workspace packages (robot-gym, grpc-workspace)
```

## Common Commands & Patterns

### Training Commands
```bash
# Verify argument structure (safe test)
uv run python lbm2/main.py --help

# Use examples as templates for actual training
# See examples/training/ for working command patterns
cat examples/training/llm_11m.sh

# Basic pattern (requires valid datasets):
uv run python lbm2/main.py \
  --model.type transformer \
  --model "include lbm2/config_presets/models/transformer_11m.yaml" \
  --data.type text \
  --total_train_samples 1000
```

### Configuration Patterns
- **Nested args**: `--model.hidden_dim 512 --data.seq_len 2048`
- **YAML inclusion**: `--model "include path/to/config.yaml"`
- **Dynamic selection**: Auto-detects subclass based on provided args

## Known Issues & Troubleshooting

### Common Failures
1. **HuggingFace Connection Errors**: Tests may fail if internet/HF access blocked
   - **Solution**: Run offline tests: `uv run pytest tests/models/ tests/params/`

2. **Memory Issues**: Large model tests may fail on limited hardware
   - **Solution**: Use smaller configs from `config_presets/models/transformer_tiny.yaml`

3. **Import Errors**: Missing dependencies or wrong Python version
   - **Solution**: Ensure `uv sync --frozen` completed successfully

4. **Training Config Validation**: Complex argument validation may fail without proper dataset paths
   - **Solution**: Use examples from `examples/training/` as templates, ensure valid manifest paths

### Environment Requirements
- **Secrets**: Create `secrets.env` with `WANDB_API_KEY` and `HF_TOKEN` for full functionality
- **S3 Access**: Required for datasets, may need AWS credentials for data loading
- **GPU**: Optional for local development, required for actual training

## Validation Steps for Code Changes

1. **Pre-commit checks**:
```bash
uv run ruff check . && uv run ruff format --check .
```

2. **Core functionality tests**:
```bash
uv run pytest tests/models/ tests/params/ -x --tb=short
```

3. **Import verification**:
```bash
uv run python -c "from lbm2.models import create_model; print('Models OK')"
uv run python -c "from lbm2.data.dataloader import get_wds_dataloader; print('Data OK')"
```

4. **Argument parsing test**:
```bash
uv run python lbm2/main.py --help | head -5
```

## Time Expectations
- **Environment setup**: 3-5 minutes (downloading dependencies)
- **Full linting**: <15 seconds  
- **Model/params tests**: ~20 seconds
- **Full test suite**: ~60 seconds (may timeout on internet-dependent tests)

## Critical Notes
- **ALWAYS use `uv run` prefix** for Python commands to ensure proper environment
- **Configuration is immutable** by design - use `object.__setattr__` if modification needed
- **Test data**: Use `tests/shared/tiny_dataset/` for tests requiring data files
- **Distributed training**: Use FSDP (`--distributed.fsdp True`) for multi-GPU
- **Command precedence**: CLI args > YAML presets > defaults

**Trust these instructions** - they are validated and current. Only search for additional details if these instructions are incomplete or commands fail with documented alternatives.