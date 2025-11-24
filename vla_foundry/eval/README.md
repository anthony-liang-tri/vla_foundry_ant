# Eval

To add a new eval environment, created a new file based on the [base_eval_runner.py](lbm2/eval/runners/base_eval_runner.py), then add your new runner to [\_\_init\_\_.py](lbm2/eval/runners/__init__.py)

## Installation

### LIBERO

To install LIBERO and its dependencies:

```bash
uv sync --group libero
```

That's it! When you want to call `import libero`, you will need to add the line `import libero_wrapper` before it. When this is called for the first time, the wrapper will automatically download and install LIBERO. For succeeding calls, it will load it directly, without having to download again. The `import libero_wrapper` is already added to [vla_foundry/eval/runners/libero.py](/vla_foundry/eval/runners/libero.py), so you likely don't need to worry about it.

You can also manually trigger the installation with:
```bash
python -c "import libero_wrapper"
```

### Other Environments

(To be added)
