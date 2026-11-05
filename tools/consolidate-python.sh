#!/usr/bin/env bash
#
# consolidate-python.sh -- merge the four absorbed Python wrapper repositories
# into this one, preserving history, deterministically and repeatably.
#
# ############################################################################
# #                                                                          #
# #   WARNING -- RUN git filter-repo EXACTLY ONCE PER SOURCE REPOSITORY,     #
# #   OVER `main` ONLY, AND NEVER RE-RUN IT WITH VARIED RULES AGAINST A      #
# #   TREE THAT HAS ALREADY BEEN REWRITTEN.                                  #
# #                                                                          #
# #   This is the Python-side sibling of the C++ `tools/consolidate.sh` in   #
# #   fastfields-lib, and it is deliberately the simpler case: none of the   #
# #   four absorbed repos has a second long-lived branch that has to stay    #
# #   mergeable (the C++ side had `teeny`), so there is no shared-ancestor   #
# #   invariant to preserve across refs. Rule drift is therefore merely      #
# #   wrong, not fatal -- but it is still wrong. If the rules must change,   #
# #   throw away every rewritten ref and re-run from pristine clones.        #
# #                                                                          #
# ############################################################################
#
# What it does
# ------------
#   1. clone the four absorbed repos pristine (or reuse --src clones)
#   2. per repo: ONE `git filter-repo --refs main` pass that moves paths into
#      the consolidated layout. Path moves only -- nothing is dropped, because
#      a dropped path turns the commits that only touched it into empty
#      commits, which filter-repo prunes: that is how a history-preserving
#      merge quietly loses history. Files with no place in the merged tree are
#      parked under docs/legacy/<sub>/ or ci/legacy/<sub>/ and deleted
#      afterwards by an ordinary commit, which keeps them reachable.
#   3. merge the four rewritten `main`s into THIS repo's history
#      (`--allow-unrelated-histories`; the four path sets are disjoint from
#      each other and from this repo's by construction, so the merges cannot
#      conflict)
#
# This repo (`fastfields`) is NOT rewritten. It hosts the merged distribution
# because it already owns the `fastfields` distribution name, and leaving its
# history untouched means `main` stays a true ancestor of the result: every
# existing commit SHA, permalink and open PR against it survives.
#
# Target layout
# -------------
#   fastfields/auto/        this repo (untouched)
#   fastfields/helpers/     fastfields-helpers
#   fastfields/numpy/       fastfields-numpy
#   fastfields/cupy/        fastfields-cupy
#   fastfields/torch/       fastfields-torch
#   tests/                  all five suites, flat, with unique module names
#   docs/<sub>.md           each absorbed repo's docs landing page
#   docs/api/<sub>.md       each absorbed repo's API reference stub
#   docs/legacy/<sub>/      parked: pyproject, README, LICENSE, zensical, ...
#   ci/legacy/<sub>/        parked: the absorbed repos' workflows
#
# The `fastfields/<sub>/` package directories need no path rewriting at all --
# every absorbed repo already ships its subpackage at its final path, because
# they were always PEP 420 portions of one `fastfields` namespace. That is the
# property that makes this merge cheap, and it is also the property the merged
# distribution must not break: there is no `fastfields/__init__.py` anywhere,
# and creating one would shadow `fastfields.dlpack`, which ships separately.
#
# The pre-namespace flat layouts (`fastfields_numpy/`, `fastfields_cupy/`,
# `fastfields_torch/`) are mapped onto `fastfields/<sub>/` rather than parked,
# so `git log --follow` crosses that old rename unbroken. Verified safe: no
# commit in any of the three repos contains both layouts at once.
#
# Usage
#   tools/consolidate-python.sh --out DIR [--src DIR]
#
#   --out DIR     where the consolidated clone is created (wiped if it exists)
#   --src DIR     directory holding/receiving the pristine clones
#                 (default: <out>/../src)
#   -h|--help     this text
#
# Requires: git, git-filter-repo (pip install git-filter-repo), python3.

set -euo pipefail

die() { printf 'consolidate-python: %s\n' "$*" >&2; exit 2; }
say() { printf '\n=== %s\n' "$*" >&2; }

GITHUB="https://github.com/fastfields"
HOST="fastfields"
SUBS="helpers numpy cupy torch"

OUT=""; SRC=""
while [ $# -gt 0 ]; do
    case "$1" in
        --out)     OUT="${2:-}"; shift 2 ;;
        --src)     SRC="${2:-}"; shift 2 ;;
        -h|--help) sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | head -n -1; exit 0 ;;
        *)         die "unknown argument: $1 (try --help)" ;;
    esac
done
[ -n "$OUT" ] || die "--out is required (try --help)"

command -v git-filter-repo >/dev/null || die "git-filter-repo not found (pip install git-filter-repo)"
command -v python3         >/dev/null || die "python3 not found"

mkdir -p -- "$OUT"; OUT="$(cd -- "$OUT" && pwd)"
[ -n "$SRC" ] || SRC="$(dirname -- "$OUT")/src"
mkdir -p -- "$SRC"; SRC="$(cd -- "$SRC" && pwd)"

# ---------------------------------------------------------------------------
# 1. pristine clones
# ---------------------------------------------------------------------------
say "stage 1: pristine clones in $SRC"
for s in $SUBS; do
    r="fastfields-$s"
    if [ -d "$SRC/$r/.git" ]; then printf '  reuse %s\n' "$r" >&2
    else git clone --quiet "$GITHUB/$r" "$SRC/$r" || die "clone failed: $r"; fi
done
if [ -d "$SRC/$HOST/.git" ]; then printf '  reuse %s\n' "$HOST" >&2
else git clone --quiet "$GITHUB/$HOST" "$SRC/$HOST" || die "clone failed: $HOST"; fi

# ---------------------------------------------------------------------------
# 2. path rewrite -- ONE filter-repo pass per absorbed repo, --refs main
# ---------------------------------------------------------------------------
rules_for() {
    # $1 = subpackage name (helpers|numpy|cupy|torch)
    local s="$1"
    cat <<PY
f = filename.decode('utf-8', 'surrogateescape')

# the subpackage itself: already at its final path
if f.startswith('fastfields/$s/'):
    return filename

# the pre-namespace flat layout, folded onto the same final path so that
# git log --follow crosses the rename
if f.startswith('fastfields_$s/'):
    return ('fastfields/$s/' + f[len('fastfields_$s/'):]).encode()

# tests merge into one flat suite, so every module name must be unique:
# test_$s.py already is; anything else gets a _$s suffix
# (tests/test_reg_family_routing.py exists in three of the four repos).
if f.startswith('tests/'):
    base = f[len('tests/'):]
    if base == 'test_$s.py':
        return filename
    if base.endswith('.py'):
        return ('tests/' + base[:-3] + '_$s.py').encode()
    return ('tests/$s-' + base).encode()

# docs: the per-repo landing page becomes a per-subpackage guide page, and
# the per-repo API stub becomes a per-subpackage API page
if f == 'docs/index.md':
    return b'docs/$s.md'
if f == 'docs/api/index.md':
    return b'docs/api/$s.md'

# everything else is parked, never dropped -- see the header
if f.startswith('.github/'):
    return ('ci/legacy/$s/' + f[len('.github/'):]).encode()
return ('docs/legacy/$s/' + f).encode()
PY
}

say "stage 2: path rewrite (one filter-repo pass per repo, --refs main)"
RW="$OUT/.rewritten"
rm -rf -- "${RW:?}"; mkdir -p -- "$RW"
for s in $SUBS; do
    r="fastfields-$s"
    printf '  %s\n' "$r" >&2
    rm -rf -- "${RW:?}/${r:?}"
    git clone --quiet --no-local --bare "$SRC/$r" "$RW/$r" >/dev/null
    # Keep only main, so no stray branch carries an un-rewritten copy of the
    # same commits. grep -v exits 1 on no match, which under `set -e` would
    # abort before any repo is rewritten -- deleting no refs is correct here.
    git -C "$RW/$r" for-each-ref --format='%(refname)' \
        | { grep -vE '^refs/heads/main$' || true; } \
        | while read -r ref; do git -C "$RW/$r" update-ref -d "$ref"; done
    rules_for "$s" > "$RW/$r.rules.py"
    ( cd "$RW/$r" && git filter-repo --force --refs main \
        --filename-callback "$(cat "$RW/$r.rules.py")" >/dev/null )
done

# ---------------------------------------------------------------------------
# 3. merge the four rewritten mains into the host repo
# ---------------------------------------------------------------------------
say "stage 3: merge into $HOST"
rm -rf -- "${OUT:?}/repo"
git clone --quiet "$SRC/$HOST" "$OUT/repo"
cd "$OUT/repo"
git checkout --quiet -B consolidate-python origin/main

for s in $SUBS; do
    r="fastfields-$s"
    git remote add "$r" "$RW/$r"
    git fetch --quiet "$r" "refs/heads/main:refs/heads/src/$s/main"
    git merge --quiet --allow-unrelated-histories --no-edit \
        -m "Merge fastfields-$s history into the fastfields monorepo" \
        "src/$s/main" \
        || die "merge of $r conflicted -- the path rule sets are not disjoint"
done

# ---------------------------------------------------------------------------
# 4. checks
# ---------------------------------------------------------------------------
say "stage 4: checks"
fail=0
for s in $SUBS; do
    n="$(git ls-tree -r --name-only HEAD | grep -c "^fastfields/$s/" || true)"
    [ "$n" -gt 0 ] || { printf '  FAIL fastfields/%s/ is empty\n' "$s" >&2; fail=1; }
done
# The whole point of the exercise: no __init__.py directly under fastfields/.
if git ls-tree -r --name-only HEAD | grep -qx 'fastfields/__init__.py'; then
    printf '  FAIL fastfields/__init__.py exists -- the PEP 420 namespace is broken\n' >&2
    fail=1
fi
# Duplicate test module basenames would make pytest refuse to collect.
dups="$(git ls-tree -r --name-only HEAD | sed -n 's|^tests/\(test_.*\.py\)$|\1|p' \
        | sort | uniq -d)"
[ -z "$dups" ] || { printf '  FAIL duplicate test module(s): %s\n' "$dups" >&2; fail=1; }
[ "$fail" -eq 0 ] || die "post-merge checks failed"
printf '  ok\n' >&2

say "done: $OUT/repo (branch consolidate-python)"
