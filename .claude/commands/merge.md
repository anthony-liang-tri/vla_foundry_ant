Merge a branch into the current branch and resolve any conflicts.

The argument is the branch to merge, e.g. `/merge main` or `/merge origin/main`

Steps:
1. Run `git fetch origin` to get latest remote state.
2. Run `git status` to confirm the working tree is clean. If not, warn the user and stop.
3. Run `git merge <branch>` (the argument).
4. If the merge succeeds cleanly, report success and the merge commit.
5. If there are conflicts:
   a. List all conflicting files with `git diff --name-only --diff-filter=U`.
   b. For each conflicting file:
      - Read the file and understand both sides of the conflict (HEAD vs incoming).
      - Analyze the intent of each change — don't just pick one side blindly.
      - Edit the file to produce a correct resolution that preserves both intents where possible.
      - Run `git add <file>` after resolving.
   c. After all conflicts are resolved, run `uv run ruff check --fix vla_foundry/` to fix any lint issues introduced.
   d. Create the merge commit with a descriptive message noting which conflicts were resolved.
   e. Show a summary of what was resolved and how.

If no argument is provided, default to merging `origin/main`.