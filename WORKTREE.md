# Git Worktree Guide for ACT

This project is a good fit for `git worktree` because training and evaluation work often run for a long time, while code changes continue in parallel.

## Why use worktrees here

- Run multiple branches at the same time without repeated clone/install steps.
- Keep one clean `main` tree for quick fixes and sync.
- Isolate experiments (for example: robustness changes vs. policy changes).

## Suggested directory layout

Assume the main repository is:

```bash
/Volumes/Elements/act
```

Create sibling worktrees:

```bash
mkdir -p /Volumes/Elements/act-worktrees
```

Example result:

```text
/Volumes/Elements/act                 # main checkout
/Volumes/Elements/act-worktrees/robustness-exp
/Volumes/Elements/act-worktrees/policy-refactor
```

## Create a new worktree from `origin/main`

Run in the main repo (`/Volumes/Elements/act`):

```bash
git fetch origin
git worktree add /Volumes/Elements/act-worktrees/robustness-exp \
  -b feature/robustness-exp origin/main
```

This does 3 things:

- Creates branch `feature/robustness-exp`.
- Creates a new checkout at `/Volumes/Elements/act-worktrees/robustness-exp`.
- Sets branch start point from `origin/main`.

## Add a worktree for an existing local branch

```bash
git worktree add /Volumes/Elements/act-worktrees/my-branch my-branch
```

## Daily commands

List all worktrees:

```bash
git worktree list
```

Update main branch in main checkout:

```bash
cd /Volumes/Elements/act
git pull --ff-only
```

Rebase a feature worktree on latest main:

```bash
cd /Volumes/Elements/act-worktrees/robustness-exp
git fetch origin
git rebase origin/main
```

## Remove a finished worktree

```bash
git worktree remove /Volumes/Elements/act-worktrees/robustness-exp
git branch -d feature/robustness-exp
git worktree prune
```

If the branch is merged remotely and local deletion fails, use `git branch -D <branch>` only after confirming no unmerged work is needed.

## Repo-specific notes

- Keep large training outputs out of git history; store under ignored output directories.
- Reuse one shared conda env (`aloha`) across worktrees.
- Prefer one topic branch per worktree. Avoid using one branch in multiple worktrees.

## Quick workflow template

```bash
# 1) Create isolated branch/worktree
cd /Volumes/Elements/act
git fetch origin
git worktree add /Volumes/Elements/act-worktrees/<topic> -b feature/<topic> origin/main

# 2) Work normally
cd /Volumes/Elements/act-worktrees/<topic>
# edit / run training / commit

# 3) Keep branch current
git fetch origin
git rebase origin/main

# 4) Merge via PR, then cleanup
cd /Volumes/Elements/act
git worktree remove /Volumes/Elements/act-worktrees/<topic>
git branch -d feature/<topic>
git worktree prune
```
