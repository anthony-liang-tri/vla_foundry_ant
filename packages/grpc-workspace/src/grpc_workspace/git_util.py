"""
Contains functions for git metadata retrieval.
"""

import warnings

import git


def maybe_get_current_commit_sha(default: str | None = None) -> str | None:
    """
    Attempts to find and return the git SHA of the current HEAD commit if the
    current working directory is contained in a git repository. In the case
    that the current working directory is not in a git repository, returns the
    supplied default string, or None.

    Keyword argument:

    default -- A supplied default string or None to be returned in the case
               that the current working directory is not contained in a git
               repository.
    """
    git_sha = default
    try:
        repo = git.Repo(search_parent_directories=True)
        git_sha = repo.head.object.hexsha
    except git.InvalidGitRepositoryError:
        warnings.warn(
            "\nWarning: Cannot load git repository information. This can "
            "happen when get_current_commit_sha() is called from "
            "within a Bazel test environment.",
            stacklevel=2,
        )
    return git_sha


def maybe_get_current_branch_name(default: str | None = None) -> str | None:
    """
    Attempts to find and return the name of the current active git branch if
    the current working directory is contained in a git repository. In the case
    that the current working directory is not in a git repository, returns the
    supplied default string, or None.

    Keyword argument:

    default -- A supplied default string or None to be returned in the case
               that the current working directory is not contained in a git
               repository.
    """
    branch_name = default
    try:
        repo = git.Repo(search_parent_directories=True)
        branch_name = repo.active_branch.name
    except git.InvalidGitRepositoryError:
        warnings.warn(
            "\nWarning: Cannot load git repository information. This can "
            "happen when maybe_get_current_branch_name() is called from "
            "within a Bazel test environment.",
            stacklevel=2,
        )
    return branch_name


def maybe_get_latest_main_commit_sha(default: str | None = None) -> str | None:
    """
    Attempts to find and return the git SHA of the latest commit on the main
    branch. If the current working directory is not contained in a git
    repository, returns the supplied default string, or None.

    Keyword arguments:

    default -- A supplied default string or None to be returned in the case
               that the current working directory is not contained in a git
               repository
    """
    latest_main_sha = default
    try:
        repo = git.Repo(search_parent_directories=True)
        latest_main_sha = repo.commit("main").hexsha
    except (git.InvalidGitRepositoryError, git.BadName):
        warnings.warn(
            "\nWarning: Cannot load git repository information or find "
            "the main branch. This can happen when "
            "maybe_get_latest_main_commit_sha() is called from within a "
            "Bazel test environment or the main branch does not exist.",
            stacklevel=2,
        )
    return latest_main_sha


def maybe_get_remote_url_from_active_branch(
    default: str | None = None,
) -> str | None:
    """
    Attempts to find the git remote URL from the current active branch's
    tracked branch. The default argument is returned in the following cases:
    - the current working directory is not contained in a git repository,
    - the active branch doesn't have a remote tracking branch, or
    - the active branch's remote tracking branch lacks a remote name.

    Keyword argument:

    default -- A supplied default string or None to be returned in case the
               remote URL is not detected.
    """
    remote_url = default
    try:
        repo = git.Repo(search_parent_directories=True)
        remote_tracking_branch = repo.active_branch.tracking_branch()
        if remote_tracking_branch and remote_tracking_branch.remote_name:
            for remote in repo.remotes:
                if remote.name == remote_tracking_branch.remote_name:
                    remote_url = remote.url
                    break
            else:
                warnings.warn(
                    "\nWarning: Cannot detect a remote URL for the remote "
                    "branch tracking the active branch.\n"
                    f"Defaulting to use the remote URL {default}.\n"
                    "See the Git Pro book for more details on working with "
                    "remotes and adding a tracked branch:\n"
                    "https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes",
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "\nWarning: Cannot detect a remote branch tracking the active "
                f"branch.\nDefaulting to use the remote URL {default}.\n"
                "Consider using the command `git branch "
                "--set-upstream-to=your_remote/branch_name`.\n"
                "See the Git Pro book for more details on setting up a "
                "tracked branch:\n"
                "https://git-scm.com/book/en/v2/Git-Branching-Remote-Branches#_tracking_branches",
                stacklevel=2,
            )
    except git.InvalidGitRepositoryError:
        warnings.warn(
            "\nWarning: Cannot load git repository information. This can "
            "happen when maybe_get_remote_url_from_active_branch() is called "
            "from within a Bazel test environment.\n"
            f"Defaulting to use the remote URL {default}.",
            stacklevel=2,
        )
    return remote_url
