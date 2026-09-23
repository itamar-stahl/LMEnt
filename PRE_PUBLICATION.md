# Pre-publication checklist

This repository is private. These are the steps to complete **before** its
visibility is changed to public. Each one is cheap to do while the repo is
private and expensive or impossible to undo afterwards.

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
