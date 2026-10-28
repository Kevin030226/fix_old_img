#!/usr/bin/env bash
# Release-tag guard: prove that a tag push is publishing the right annotated tag.
#
# Why this is not an inline check
# -------------------------------
# actions/checkout@v5 checks a tag push out by fetching the tag object and then
# force-updating refs/tags/<tag> to the peeled commit. By the time a job runs, the
# checkout-owned ref is a lightweight ref pointing straight at the commit, so
# asking it what type it is -- `git cat-file -t` -- answers "commit", and asking it
# for its annotation -- `git tag -l --format='%(contents)'` -- yields the commit
# subject instead. A correct annotated tag therefore looks unannotated.
#
# This guard never reads that ref. It fetches the remote tag into a namespace of
# its own, refs/release-tags/<tag>, and validates that private copy: the object
# type, the peeled commit, and the annotation used for the release title.
#
# Usage (sourced, so callers can test it and reuse the validated values)
# ---------------------------------------------------------------------
#   source scripts/ci/release_tag_guard.sh
#   release_tag_guard "$GITHUB_REF_NAME" "$main_sha"      # validates, or dies
#   release_tag_write_outputs                             # -> $GITHUB_OUTPUT
#
# On success release_tag_guard sets two globals:
#   RELEASE_TAG_SHA    peeled commit of the annotated tag, lowercase hex
#   RELEASE_TAG_TITLE  single-line release title from the tag annotation
# Set as globals rather than printed, so the title is never re-parsed by the
# caller and can never be word-split or accidentally evaluated.

set -euo pipefail

#: A plain vMAJOR.MINOR.PATCH name is required before anything reaches a refspec.
RELEASE_TAG_RE='^v[0-9]+\.[0-9]+\.[0-9]+$'
#: A commit id must be a full lowercase SHA-1; short or uppercase forms are refused.
RELEASE_SHA_RE='^[0-9a-f]{40}$'
#: Namespace owned by this guard. actions/checkout never writes it.
RELEASE_TAG_NS='refs/release-tags'
#: UTF-8 byte budget for the title. Exceeding it is an error, never a truncation:
#: a silently shortened title is a wrong title.
RELEASE_TITLE_MAX=200

RELEASE_TAG_SHA=''
RELEASE_TAG_TITLE=''

_release_tag_die() {
  printf '::error::%s\n' "$*" >&2
  return 1
}

# release_tag_guard <tag> <expected_commit_sha> [remote]
release_tag_guard() {
  local tag=${1:-}
  local expected=${2:-}
  local remote=${3:-origin}
  local private sha peeled main_sha title bytes

  RELEASE_TAG_SHA=''
  RELEASE_TAG_TITLE=''

  if [[ ! $tag =~ $RELEASE_TAG_RE ]]; then
    _release_tag_die "'$tag' is not a vMAJOR.MINOR.PATCH tag name; refusing to use it in a refspec"
    return 1
  fi
  if [[ ! $expected =~ $RELEASE_SHA_RE ]]; then
    _release_tag_die "expected commit '$expected' is not a 40-character lowercase hex SHA"
    return 1
  fi

  private="$RELEASE_TAG_NS/$tag"
  if git rev-parse --quiet --verify "$private" >/dev/null 2>&1; then
    _release_tag_die "$private already exists; refusing to reuse a stale private ref"
    return 1
  fi

  # No leading '+': the private ref is created by this fetch and must never be
  # force-updated over a leftover object. stderr is dropped so no transport or
  # credential detail can reach the log; the message below is the whole story.
  if ! git fetch --no-tags --quiet "$remote" "refs/tags/$tag:$private" 2>/dev/null; then
    _release_tag_die "could not fetch tag '$tag' from '$remote'; is it pushed?"
    return 1
  fi

  if [ "$(git cat-file -t "$private" 2>/dev/null || true)" != 'tag' ]; then
    _release_tag_die "$tag is not an annotated tag; the release title comes from its annotation"
    return 1
  fi
  peeled="$(git rev-parse --quiet --verify "$private^{commit}" 2>/dev/null || true)"
  if [[ ! $peeled =~ $RELEASE_SHA_RE ]]; then
    _release_tag_die "$tag has no commit target; a tag object must point at a commit"
    return 1
  fi
  if [ "$peeled" != "$expected" ]; then
    _release_tag_die "$tag points at $peeled, but the expected commit is $expected"
    return 1
  fi
  # Exact commit, not merely an ancestor: a tag on an older commit would publish
  # code that main no longer builds.
  main_sha="$(git rev-parse --quiet --verify refs/remotes/origin/main 2>/dev/null || true)"
  if [[ ! $main_sha =~ $RELEASE_SHA_RE ]]; then
    _release_tag_die 'refs/remotes/origin/main is not available; cannot prove the tag is on main'
    return 1
  fi
  if [ "$peeled" != "$main_sha" ]; then
    _release_tag_die "$tag points at $peeled, but origin/main is $main_sha"
    return 1
  fi

  # Read the annotation from the private tag object. %(contents:subject) is the
  # first paragraph with newlines folded to spaces, which is what a one-line
  # Release title needs. The public ref is never consulted.
  title="$(git for-each-ref --format='%(contents:subject)' "$private")"
  if [ -z "${title//[[:space:]]/}" ]; then
    _release_tag_die "$tag has an empty annotation; the release title would be blank"
    return 1
  fi
  # A control character here means either a malformed annotation or an attempt to
  # smuggle a second line into GITHUB_OUTPUT. Refuse rather than sanitise.
  if [[ $title == *[[:cntrl:]]* ]]; then
    _release_tag_die "$tag annotation contains a control character (CR, LF, tab or similar); refusing to publish it as a title"
    return 1
  fi
  bytes=$(printf '%s' "$title" | wc -c | tr -d ' ')
  if [ "$bytes" -gt "$RELEASE_TITLE_MAX" ]; then
    _release_tag_die "$tag annotation is $bytes bytes; the release title budget is $RELEASE_TITLE_MAX"
    return 1
  fi

  RELEASE_TAG_SHA=$peeled
  RELEASE_TAG_TITLE=$title
  printf 'publishable tag %s at %s\n' "$tag" "$peeled"
  return 0
}

# release_tag_write_outputs
#
# Emit the validated values. Both are guaranteed single-line and control-free by
# release_tag_guard, so the plain key=value form cannot corrupt the output file.
release_tag_write_outputs() {
  local target=${GITHUB_OUTPUT:-}
  if [ -z "$RELEASE_TAG_SHA" ] || [ -z "$RELEASE_TAG_TITLE" ]; then
    _release_tag_die 'release_tag_guard has not produced validated values yet'
    return 1
  fi
  if [ -z "$target" ]; then
    _release_tag_die 'GITHUB_OUTPUT is not set; refusing to drop the validated values'
    return 1
  fi
  {
    printf 'tag_sha=%s\n' "$RELEASE_TAG_SHA"
    printf 'release_title=%s\n' "$RELEASE_TAG_TITLE"
  } >> "$target"
}
