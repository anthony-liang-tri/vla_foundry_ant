Resolve all current git merge/rebase conflicts in the working tree.

This command is for when you already have unresolved conflicts (e.g. from a merge or rebase done outside Claude).

Steps:
1. Run `git status` to confirm we're in a conflict state.
2. List conflicting files with `git diff --name-only --diff-filter=U`.
3. For each conflicting file:
   - Read the full file.
   - Identify all conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`).
   - Understand the intent of both sides by reading surrounding context and git log for recent changes to that file.
   - Resolve intelligently — preserve both intents where possible, or pick the correct side with justification.
   - Remove all conflict markers.
   - Stage the resolved file with `git add`.
4. Run `uv run ruff check --fix vla_foundry/` to clean up any lint issues.
5. Show a summary of resolutions and ask the user to confirm before committing.
6. If confirmed, create the merge/continue the rebase.