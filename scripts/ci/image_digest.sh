#!/usr/bin/env bash
# Digest helpers for the release workflow.
#
# Why this file exists: the release workflow must prove that a promoted tag
# resolves to the digest it staged. An earlier version parsed
# `imagetools inspect --format '{{json .Manifest}}'` with sed; that output is
# pretty-printed JSON, so the regex never matched and the promotion step could
# only fail. It also produced one match per manifest in an index, so even a
# whitespace-tolerant regex could compare several digests against one.
#
# The rule here is deliberately narrow:
#   * a digest is exactly `sha256:` + 64 lowercase hex characters, nothing else;
#   * the digest is read from the *raw* manifest bytes the registry serves and
#     hashed locally, which is how a registry computes the digest in the first
#     place (no template-field guessing, no JSON parsing);
#   * a read must yield exactly one value, or it fails.
#
# This file is a sourced library: it must not install global traps, must not
# leave files behind on any exit path (including SIGINT/SIGTERM), and must not
# depend on the caller enabling `pipefail`.
#
# Usage: `. scripts/ci/image_digest.sh`

#: Strict shape. Uppercase is rejected on purpose: registry digests are
#: lowercase hex, so accepting uppercase would hide a mangled value.
#: Not `readonly`: the library may legitimately be sourced twice in one shell
#: (a re-source must not abort the caller with "readonly variable").
DIGEST_RE='^sha256:[0-9a-f]{64}$'

# assert_sha256_digest <value> <label>
# Fails (exit 1) unless <value> is exactly one well-formed sha256 digest.
assert_sha256_digest() {
  local value="${1-}"
  local label="${2:-digest}"
  if [ -z "$value" ]; then
    echo "::error::$label is empty" >&2
    return 1
  fi
  # Reject embedded whitespace explicitly: a value with a trailing newline or a
  # space would otherwise be caught only by luck of the regex engine.
  case "$value" in
    *[[:space:]]*)
      echo "::error::$label contains whitespace: '$value'" >&2
      return 1
      ;;
  esac
  if ! printf '%s' "$value" | grep -Eq "$DIGEST_RE"; then
    echo "::error::$label is not sha256:<64 lowercase hex>: '$value'" >&2
    return 1
  fi
  return 0
}

# require_buildx: fail early, with a message that names the tool, rather than
# letting a promotion step fail later for an unrelated-looking reason.
require_buildx() {
  if ! docker buildx version >/dev/null 2>&1; then
    echo "::error::docker buildx is not available; cannot promote or verify image digests" >&2
    return 1
  fi
  if ! docker buildx inspect --bootstrap >/dev/null 2>&1; then
    echo "::error::docker buildx could not bootstrap a builder; cannot read registry manifests" >&2
    return 1
  fi
  return 0
}

# _digest_extract_hash <sha256sum-output>
# Pulls the hash field out of `sha256sum` output without a pipeline.
# Contract: the caller has already captured the tool's exit status; a malformed
# or empty line yields an empty string, which the caller treats as failure. This
# keeps the helper correct even when the caller did not enable `pipefail`.
_digest_extract_hash() {
  local line="${1-}"
  # sha256sum prints "<64 hex>  <path>"; the hash is the first whitespace run.
  line="${line%%[[:space:]]*}"
  printf '%s' "$line"
}

# read_top_level_digest <ref>
# Prints exactly one `sha256:<64 hex>` value: the digest of the raw manifest or
# index bytes the registry returns for <ref>.
#
# Everything runs inside a subshell so that:
#   * the scratch file is private to this call (repeat and concurrent calls
#     cannot collide, and the caller's variables are untouched);
#   * the cleanup and signal traps are local to this call, so sourcing this
#     library never overwrites a trap the caller installed.
read_top_level_digest() {
  (
    set -u
    ref="$1"
    tmp=''
    cleanup() {
      if [ -n "$tmp" ] && [ -e "$tmp" ]; then
        rm -f -- "$tmp"
      fi
      tmp=''
    }
    # Cover every way this subshell can end: normal return, explicit failure,
    # and the two signals a cancelled CI job delivers.
    #
    # Signal contract: bash runs a trap when the currently blocking command
    # returns, so a SIGTERM that arrives during `imagetools` or `sha256sum` is
    # handled as soon as that command is done, then the shell exits 143 (or 130
    # for INT). The file is therefore removed on every path; it is not removed
    # instantaneously mid-syscall, which no shell can guarantee. The signal is
    # never swallowed: the subshell still exits non-zero with the conventional
    # status, so a cancelled job cannot go on to promote an image.
    trap 'cleanup' EXIT
    trap 'cleanup; exit 130' INT
    trap 'cleanup; exit 143' TERM

    if ! tmp=$(mktemp "${TMPDIR:-/tmp}/fiximg-digest.XXXXXXXX"); then
      tmp=''
      echo "::error::could not create a scratch file to read the manifest for $ref" >&2
      exit 1
    fi
    # stderr is deliberately NOT redirected: a registry or auth failure must
    # stay visible in the job log.
    if ! docker buildx imagetools inspect --raw "$ref" > "$tmp"; then
      echo "::error::could not read the raw manifest for $ref" >&2
      exit 1
    fi
    if [ ! -s "$tmp" ]; then
      echo "::error::registry returned an empty manifest for $ref" >&2
      exit 1
    fi
    # sha256 over the file's bytes is the digest the registry itself assigns.
    # The tool's exit status is captured explicitly instead of relying on a
    # pipeline's last command, so this is correct without `pipefail`.
    if ! sum=$(sha256sum "$tmp"); then
      echo "::error::could not hash the manifest for $ref" >&2
      exit 1
    fi
    hash=$(_digest_extract_hash "$sum")
    if [ -z "$hash" ]; then
      echo "::error::hashing the manifest for $ref produced no digest" >&2
      exit 1
    fi
    digest="sha256:$hash"
    if ! assert_sha256_digest "$digest" "digest read back for $ref"; then
      exit 1
    fi
    printf '%s' "$digest"
  )
}

# verify_promoted_digest <ref> <expected>
# Reads <ref> back and requires it to equal <expected> exactly.
verify_promoted_digest() {
  local ref="$1"
  local expected="$2"
  assert_sha256_digest "$expected" "staged digest" || return 1
  local published
  if ! published=$(read_top_level_digest "$ref"); then
    return 1
  fi
  if [ "$published" != "$expected" ]; then
    echo "::error::$ref resolves to $published, expected $expected" >&2
    return 1
  fi
  echo "promoted $ref -> $published"
}
