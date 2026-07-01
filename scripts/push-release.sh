#!/usr/bin/env bash
# Guarded release-tag push.
#
# Mirrors the CI repo-guard in .github/workflows/publish.yml ("Verify tag is
# reachable from master"): a release tag is ONLY published if its commit is an
# ancestor of the canonical master. Pushing a tag that is not yet on master
# creates an orphan tag and the publish job refuses — this script catches that
# locally, before the push, so it cannot happen.
#
# Release order (master is branch-protected, so the tag must land AFTER master):
#   1. merge the release commit into master (via PR)
#   2. ./scripts/push-release.sh <remote> <tag>
#
# Usage: ./scripts/push-release.sh upstream v0.5.1
set -euo pipefail

remote="${1:?usage: push-release.sh <remote> <tag>}"
tag="${2:?usage: push-release.sh <remote> <tag>}"

echo "Fetching ${remote}/master …"
git fetch -q "$remote" master

tag_sha="$(git rev-parse "${tag}^{commit}")"
master_sha="$(git rev-parse "${remote}/master")"

if ! git merge-base --is-ancestor "$tag_sha" "$master_sha"; then
    echo "ERROR: tag ${tag} (${tag_sha}) is NOT reachable from ${remote}/master (${master_sha})." >&2
    echo "       Merge the release commit into master (PR) BEFORE pushing the tag —" >&2
    echo "       otherwise publish.yml's repo-guard will refuse to publish an orphan tag." >&2
    exit 1
fi

echo "OK: ${tag} is reachable from ${remote}/master — pushing tag."
git push "$remote" "$tag"
