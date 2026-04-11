
Set up a new workspace (git worktree) for feature development.

The argument is the feature name, e.g. `/new-workspace my_feature`.

Steps:
1. Fetch `origin/main` to get the latest upstream.
2. Create a new git worktree at `worktrees/jean/$ARGUMENTS` with a new branch `jean/$ARGUMENTS` based on `origin/main`.
3. Run `uv sync` inside the new worktree to install dependencies.
4. Report the path to the new workspace when done.

If no argument is provided, ask the user for a feature name before proceeding.
