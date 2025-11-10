# Eval

To add a new eval environment, created a new file based on the [base_eval_runner.py](lbm2/eval/runners/base_eval_runner.py), then add your new runner to [\_\_init\_\_.py](lbm2/eval/runners/__init__.py)

Note that the requirements for the runners here are not handled in the uv `pyproject.toml`. We have not yet figured out the uv setup for these, so we recommend installing these sim environments on a separate conda/uv environment.