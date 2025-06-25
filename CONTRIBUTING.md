# Contributing to LBM
## Development Process
1. Fork the project.
Go to [LBM2](https://github.com/TRI-ML/lbm2) and click the "fork" button to create your own copy of the project.
2. Clone your fork locally.
```
git clone git@github.com:your-username/lbm2.git
```
3. Add the upstream repository.
```
git remote add upstream git@github.com:TRI-ML/lbm2.git
git remote set-url --push upstream no_push
```
`git remote -v` should now list the upstream branch.

4. Pull the latest changes.
```
git fetch upstream
```
5. Create a branch to develop in.
```
git checkout -b my_feature_branch upstream/main
```
Make changes within this branch, committing them as you go.

6. Create a pull request.
```
git push origin my_feature_branch
```
Go to the LBM2 [repo](https://github.com/TRI-ML/lbm2).
There should be an option to create a pull request at the top,
follow instructions there to create a pull request

7. Code review.
We're using github code reviews in this codebase.

If changes are requested to the review there are a few options for
handling this. One option is to make changes in your local branch,
amend the commit, and force push to your branch.
```
# Make code changes.
git commit --amend --no-edit
git push -f origin
```

## Updating requirements
### Using `uv`
See `uv` [docs](https://docs.astral.sh/uv/concepts/projects/dependencies/) for details.

#### Adding a requirement
```bash
uv add <dep>
# For example, here's adding httpx:
uv add httpx
# Or adding httpx with the version specified:
# (This is also how you update the version).
uv add "httpx>=0.20"
```
This adds the requirement to `pyproject.toml` and updates the `uv.lock` file.
The `uv.lock` file tracks the exact versions of dependencies so that the build
is reproducible across machines.
After doing this you'll need to commit the updated `uv.lock` file to git.
A lot of other functionality in pip also exists (e.g., system dependent requirements).
Again, see the `uv` docs for details.

#### Removing a requirement
```bash
uv remove <dep>
# For example, here's removing httpx:
uv remove httpx
```
Again, this updates `pyproject.toml` and `uv.lock`.
After doing this you'll need to commit the updated `uv.lock` file to git.

#### Generating the `uv.lock` file
If the `uv.lock` file doesn't exist, it can be created by running `uv sync`.

## Guidelines
1. Code should generally have test coverage.
2. Code should generally be documented and follow the style guidelines below.
3. Code should always be reviewed by at least one feature reviewer.

### Stylistic Guidelines
We generally follow the same guidelines as
[Anzu](https://github.shared-services.aws.tri.global/robotics/anzu),
which basically follows these [amended google python style guidelines](https://drake.mit.edu/styleguide/pyguide.html).
It's helpful to set up your editor to have a hotkey for formatting python code.
We use [black](https://github.com/psf/black) for formatting code.

### Test Coverage
This codebase uses [pytest](https://docs.pytest.org/en/stable/). Pytest has many useful
features (e.g., [`@pytest.mark.parametrize`](https://docs.pytest.org/en/stable/how-to/parametrize.html)).

To run tests locally:
```bash
# Run all the tests:
pytest .

# Run a specific test:
pytest /path/to/my/test.py -s -k partial_string_in_test_name
```

### Reviewing
Generally try to follow these
[guidelines](https://google.github.io/eng-practices/review/reviewer/standard.html)
when reviewing.
