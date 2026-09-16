# Contributing to Wattbot 2026

Seven people, seven coding agents, one `main`. Agents make it cheap to write code and
expensive to review it, so every rule here exists to protect reviewer attention and keep
`main` releasable at all times.

Read this once. It should take five minutes.

## Branches

```
<yourname>/<type>-<short-slug>

saizan/feat-hybrid-retriever
priya/fix-chunk-overlap
tom/exp-rerank-ablation
```

Types: `feat`, `fix`, `exp` (experiment), `docs`, `chore`.

- Branch from the latest `main`, one branch per pull request, delete after merge.
- Nobody commits to `main` directly. Nobody force-pushes a branch someone else is on.
- Stale after 5 days? Rebase it or close it. Long-lived branches are where merge pain lives.

## Pull requests

**Target 200 lines of human-reviewed diff. Hard ceiling 400.** Past that, split it — an agent
can produce 2,000 lines in a minute and nobody reviews that honestly; they just approve it.

- One concern per PR. "Refactor + feature" is two PRs.
- Generated files (lockfiles, notebook outputs, fixtures) don't count toward the limit, but
  call them out in the description so reviewers know what to skip.
- Open it as a draft early if you want direction before the agent goes further.
- Squash-merge. The PR title becomes the commit on `main`, so write it like a commit subject.

Every PR description answers three questions:

1. **What changed and why** — the problem, not a restatement of the diff.
2. **How it was verified** — tests run, eval numbers before/after, or "manual: <what you did>".
3. **What the agent did unsupervised** — anything you didn't read line by line, flagged plainly.

Point 3 is not a confession, it's routing information. It tells the reviewer where to look hard.

## Review

- **One human approval required**, from someone who did not drive the agent that wrote it.
- **Two approvals** for: anything under `src/wattbot/config/` or `src/wattbot/index/`, prompt
  changes, dependency changes, CI changes, or anything touching a data schema.
- **Reviewer of the day** rotates weekly and picks up anything unclaimed within 24h. Being
  blocked on review for more than a day means you pinged the wrong channel, not that you wait.
- Review the diff, not the agent's summary of it. If a PR is too big to review properly, say
  "too big, please split" — that is a complete and legitimate review.
- Author merges once approved and CI is green.

## What an agent may do without asking

**Green — go ahead:**
- Write and edit code inside the feature area you're actively working in.
- Add or update tests for that code.
- Docstrings, comments, type hints, formatting, local refactors inside a file you own.

**Yellow — get a human decision first:**
- Editing shared modules (`src/wattbot/config/`, `src/wattbot/index/`, anything imported by
  three or more packages).
- Changing prompts, model IDs, or retrieval parameters — these move eval numbers silently.
- Adding, removing, or upgrading a dependency.
- Touching CI, `.github/`, or anything that changes what "green" means.
- Changing a data schema or a file format written to disk.

**Red — never, agent or human:**
- Commits to `main`, force-pushes to shared branches, rewriting someone else's history.
- Secrets, `.env`, credentials, API keys — in code, in commits, in PR descriptions.
- Deleting or rewriting eval sets and their recorded baselines.
- Deleting data or migrations to make a test pass.
- Weakening, skipping, or deleting a failing test instead of fixing the cause.

## Not colliding with each other

Two agents editing the same file is the failure mode that costs us whole afternoons.

- **Claim before you start.** Comment the files or package you're taking on the issue you're
  working, or in `#wattbot-dev`. Thirty seconds of claiming beats an hour of conflict surgery.
- **Check first:** `git fetch && git log --oneline origin/main -- <path>` plus a glance at open
  PRs. If someone's PR already touches that file, wait for it or coordinate directly.
- **Own a package.** Each area has a primary owner; changes from outside that area go through
  them. Keep shared modules thin and stable so there's less to fight over.
- **Merge fast.** Small PRs that land in a day rarely conflict. Week-old branches always do.
- **Rebase before pushing**, and run agents in separate checkouts (`git worktree`) so two of
  them never share a working tree.

## Commit messages

```
<type>(<scope>): <imperative subject, <=72 chars>

Why this change exists. What it affects that isn't obvious from the diff.
Trade-offs or alternatives rejected, if any.

Refs: #123
```

- Types match branch types: `feat`, `fix`, `exp`, `docs`, `chore`. Scope is the package
  (`retrieve`, `ingest`, `eval`, ...).
- Explain **why**. The diff already shows what — an agent-written summary of what changed is
  noise in the log.
- Note behaviour changes that don't show in the diff: eval deltas, latency, cost.
- Agent-assisted commits carry a `Co-Authored-By:` trailer for the agent. Keep the human author
  as the commit author — someone is accountable for every commit.
- No "fix", "wip", "address comments". Those are review noise, and squash-merge won't save you
  if the PR title is just as vague.

## Repo structure

Put new files where they already belong. Inventing a top-level directory needs its own PR.

```
src/wattbot/          Production code, importable package
  ingest/             Loaders, parsing, chunking
  index/              Embeddings, vector store  (shared - 2 approvals)
  retrieve/           Search, filters, reranking
  generate/           LLM calls, answer assembly
  eval/               Metrics, graders, eval harness
  api/                Service layer and endpoints
  config/             Typed settings           (shared - 2 approvals)
prompts/              Prompt text, one file per prompt, versioned
configs/              YAML run configs - data only, no code
tests/                Mirrors src/wattbot/ exactly
experiments/          <name>/<date>-<topic>/ - throwaway, never imported
notebooks/            Exploration only, <name>-<topic>.ipynb
scripts/              Ops and CLI one-offs
data/                 Gitignored except data/samples/ and schemas
docs/                 Design notes and decision records
```

**Where does this file go?**

| You're writing | It goes in |
| --- | --- |
| Something imported by the app | `src/wattbot/<package>/` |
| Something you'll run once and throw away | `experiments/<name>/` |
| A prompt string | `prompts/` — never inline in code |
| A tunable number or path | `configs/` — never hardcoded |
| A test | `tests/`, mirroring the source path |
| A new top-level directory | A PR that only adds it, plus a line here |

If it takes more than a few seconds to decide, ask in `#wattbot-dev` rather than guessing.
A wrong guess ends up copy-pasted by six other agents within a week.

## Definition of done

- [ ] Tests pass locally, and there's a new test if behaviour changed
- [ ] Diff is under 400 lines, or the PR says why it can't be
- [ ] PR describes what changed, how it was verified, and what wasn't reviewed line by line
- [ ] No secrets, no skipped tests, no unrelated files
- [ ] Eval numbers reported if retrieval or generation changed
