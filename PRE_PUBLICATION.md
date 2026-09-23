# Pre-publication checklist

This repository is private. These are the steps to complete **before** its
visibility is changed to public. Each one is cheap to do while the repo is
private and expensive or impossible to undo afterwards.

## 0. Rotate the Elasticsearch credential

`Untaught/framework/env.sh` assigns a real password for the cluster
Elasticsearch deployment as a literal. It is at the tip of every branch this
remote advertises, and it was introduced in `0cfc1fb`.

Two independent actions, which do different things:

1. **Rotate the credential on the Elasticsearch server.** This is the only step
   that removes the risk, because it makes the leaked value useless. Whether it
   can be done depends on whether that deployment is this project's to re-key or
   is shared with other groups.
2. **Remove the literal from the file.** Every consumer already reads
   `ES_PASSWORD` from the environment -- `Untaught/framework/client/es_blacklist.py`,
   `Ember-on-LMEnt/sentences_gen/blacklist_to_concept_sentences.py`, and the
   `sentences_gen` tests, which skip when it is unset. So the literal can be
   replaced with a read from an untracked local file without changing any
   consumer. Note that nine files source `Untaught/framework/env.sh` and inherit
   the variable, so they need the local file to exist before they run.

Removing it from the working tree does not remove it from history. Decide that
question together with section 1 below, since both are history questions.

## 1. Scrub non-public contributor identities from history

Git stores an author and a committer identity on every commit, and those
identities are part of the commit hash, not the working tree. A `.gitignore`
never covered them and no file diff ever showed them. Several commits in this
history carry identities that are not the contributor's public identity --
work email addresses, and names or addresses generated from a personal machine
or a login node rather than chosen (`user@some-laptop.local`,
`user@some-cluster-node`, `MACHINE\user`).

Publishing the repo publishes all of them, on every branch the remote
advertises.

### Find them

```
git log --all --format='%an <%ae>%n%cn <%ce>' | sort -u
```

Read the whole list and decide, per line, whether that identity is one the
contributor is content to publish. Ask them; do not guess on someone's behalf.
Checking only `main` is not enough -- an identity can sit on a side branch, and
a merged branch keeps its own commits.

### Rewrite them

Build a mailmap with one line per identity to replace, mapping each to the
identity that contributor wants published (a GitHub `users.noreply.github.com`
address publishes attribution without an address):

```
Proper Name <public@address> <address-to-replace>
```

Then rewrite a fresh mirror, never the working clone:

```
git clone --mirror git@github.com:itamar-stahl/LMEnt.git LMEnt.git
git -C LMEnt.git filter-repo --mailmap /path/to/mailmap
```

Verify before pushing anything -- the rewrite must change identities and
nothing else. Confirm the commit and ref counts are unchanged, and that every
branch tip still points at an identical tree:

```
git -C LMEnt.git rev-list --all --count          # must equal the original
git -C LMEnt.git for-each-ref | wc -l            # must equal the original
# and per branch, compare <branch>^{tree} against the original's
```

### Know the cost before you start

Measured on this repo, as of the commit that added this file:

- **Every commit changes hash, not just the ones being fixed.** The root commit
  is GPG-signed, `filter-repo` strips signatures, so the root changes and new
  hashes ripple through all 373 commits.
- **19 GPG signatures are destroyed**, most of them not the signatures of
  whoever runs this. Their "Verified" badges on GitHub go away and cannot be
  restored. Tell them first.
- **21 branches need force-pushing**, and every existing clone breaks. Everyone
  re-clones or hard-resets. Pick a moment when nobody has unpushed work.
- **`refs/pull/1/head` cannot be rewritten.** GitHub's pull-request refs are
  read-only, and that one contains the pre-rewrite commits. After a perfect
  force-push the old identities are still fetchable by anyone who can read the
  repo, until GitHub garbage-collects them -- which needs a request to GitHub
  Support. **A force-push alone does not finish this.** File that request and
  confirm it is done before flipping visibility.

Because of the last point, the alternative is to publish a fresh repository
built from the filtered history rather than changing this one's visibility. A
new repo has no pull-request refs and no existing clones, so it avoids every
cost above. It also discards issues, pull requests and stars, which this repo
does not yet depend on.

## 2. Stop new commits from re-adding them

A rewrite fixes history; it does not stop the next commit. Whoever commits from
a machine whose global `user.email` is not their public identity should set a
per-repository override, which leaves their other projects alone:

```
git config --local user.name  "Proper Name"
git config --local user.email "public@address"
```

Worth confirming for each contributor before publication, and worth re-checking
after, since a fresh clone does not inherit a local override.

## 3. Decide what licence covers LMEnt's own code

There is no `LICENSE` at the repository root. Without one, the 421 files of this
project's own code are all-rights-reserved on publication: readable, but not
legally reusable by anyone. That may be acceptable if the audience is graders,
who only need to read it -- a licence governs reuse, not viewing. It is not
acceptable if the paper is meant to have a usable artifact.

Two related facts, both verified:

- The only MIT licence inside this project's own tree is
  `Ember-on-LMEnt/LICENSE`, and its copyright line names EMBER's author. It came
  with that code and does not cover LMEnt.
- `Ember-on-LMEnt/external/CRISP/` and `Ember-on-LMEnt/external/snmf/` are
  vendored with **no licence text at all**. Code with no stated licence carries
  no permission to redistribute, so both need a licence confirmed with their
  authors before publication. See `THIRD_PARTY.md`.

## 4. Decide what happens to the other branches

This checklist and the reorganisation that accompanies it apply to one branch.
Publishing the repository publishes every ref the remote advertises -- 21 branch
tips plus `refs/pull/1/head` at the time of writing. Nobody has reviewed the
others, and at least one carries material that does not belong in a published
artifact: `latex_paper` has a directory whose name contains spaces
(`paper ACL Template`), a course-guidelines PDF at the root, and a stale
pre-reorganisation layout.

Make a per-branch publish / delete / keep-private decision and record it here
before visibility changes. A branch left behind also silently contradicts the
reorganisation, because the old layout stays live and reachable on it.

## 5. Open engineering items, not blockers

Recorded here so they are not lost. None of these gate publication.

- **`ai2-olmo-core==0.1.0` in both conda manifests shadows the vendored fork.**
  `environment.yml:93` and `environment.windows.yml:30` install a distribution
  whose name matches `third_party/OLMo-core/pyproject.toml`, so `site-packages`
  holds an older `olmo_core`. It is mitigated in practice: `PYTHONPATH` precedes
  `site-packages`, so sourcing `Untaught/framework/env.sh` makes the fork win.
  It only bites a script run without that. Removing the pin forces every
  contributor to rebuild their conda environment, and it is unverified whether
  `ai2-olmo==0.6.0` on the line above pulls it back in transitively -- check that
  first with `pip show ai2-olmo`.
- **`third_party/OLMo-core` carries an LMEnt-only correctness fix** (`885d87c`,
  the embedding-init ordering). It must be pushed to the fork before that tree
  could ever become a submodule. `THIRD_PARTY.md` explains the consequence.
- **`retrieval-index/create_es_index.py` still hardcodes cluster data paths**
  for its dataset inputs (the `work_dir` and the `.npy` glob). The OLMo-core
  import path was made portable; these were left, because the correct value
  depends on where a given user unpacked `LMEnt-Dataset`.
