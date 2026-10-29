#!/usr/bin/env bash
# Sanitise a release-notes file for use as a GitHub Release body.
#
# The release workflow composes the published body by copying the notes file
# from docs/releases/v<version>.md verbatim. That file is allowed to carry
# maintainer HTML comments for the repository's own readers, but those comments
# must never reach the public Release body, where they survive in the raw
# Markdown and API even though the rendered page hides them.
#
# Protocol
# --------
#   release_notes_body <notes-file>
#
# Reads the notes file as data and writes the sanitised Markdown to stdout.
# Diagnostics go to stderr only. The input is never executed, sourced, eval'd
# or interpolated; it is handed to awk through stdin redirection on a quoted
# path, so a filename full of shell metacharacters is just a filename.
#
# Contract
# --------
# Input encoding is LF-only. A carriage return anywhere in the file is rejected
# rather than reinterpreted: this repository normalises text to LF
# (.gitattributes: `* text=auto eol=lf`) and every tracked Markdown file is
# verified CR-free, so a CR byte can only mean a file the pipeline should not
# publish. Guessing -- treating 'contains a CR' as 'is uniformly CRLF' -- would
# silently delete a CR that was content and rewrite mixed endings while claiming
# byte preservation, so it is refused instead.
#
# Removed  : HTML comments (<!-- ... -->) that appear outside a literal-code
#            context, including their delimiters and any lines they occupy.
# Scope    : LF-only Markdown text. This is not a binary filter -- arbitrary
#            binary or NUL-bearing input is outside the contract, because awk
#            is not required to preserve NUL bytes.
#
# Preserved: every other text byte, in order -- headings, inline code spans,
#            fenced code blocks (backtick or tilde, any length >= 3, closed by
#            a run of the same character at least as long), visible HTML such
#            as <details>/<summary>, bare <tag> text, and LF line endings.
# Normalised: leading blank lines are dropped only when a comment was removed
#            before any visible content, i.e. when the blank lines exist purely
#            because that leading comment used to occupy them.
# Endings  : successful output is newline-terminated and never ends in an
#            unterminated partial line. An input that lacks its final terminator
#            gains exactly one LF. Existing trailing LF bytes are preserved byte
#            for byte: an input whose final bytes are three LF characters yields a
#            body whose final bytes are those same three LF characters -- the
#            helper neither trims an existing tail nor adds to one.
#
# Fail-closed exit codes (no usable output is produced):
#   2  notes file missing, not a regular file, or not readable
#   3  an HTML comment is opened but never closed
#   4  a stray --> appears outside a literal-code context
#   5  the input is empty, or sanitising leaves nothing visible
#   6  the file contains a carriage return (LF-only contract)
#   7  the carriage-return count itself could not be computed
#
# Fence semantics follow CommonMark closely enough for release notes: an
# opening fence is up to three spaces of indent followed by three or more
# backticks or tildes (a backtick fence's info string may not contain a
# backtick); it closes on a run of the same character, at least as long, with
# nothing but whitespace after it. Inside a fence, comment delimiters are
# content and are copied through untouched.

# No shell option is changed at source time: a caller that sources this file
# keeps whatever state it had, including whether nounset is enabled.

_release_notes_body_awk() {
  cat <<'AWK'
function is_ws(s,   i, c) {
  for (i = 1; i <= length(s); i++) {
    c = substr(s, i, 1)
    if (c != " " && c != "\t" && c != "\r") { return 0 }
  }
  return 1
}

function indent_width(s,   i, w, c) {
  w = 0; i = 1
  while (i <= length(s)) {
    c = substr(s, i, 1)
    if (c == " ") { w += 1; i += 1 }
    else if (c == "\t") { w += 4; i += 1 }
    else { break }
  }
  return w * 1000 + i
}

function fence_open(s,   code, n, c, i) {
  code = indent_width(s)
  if (int(code / 1000) > 3) { return "" }
  i = code % 1000
  c = substr(s, i, 1)
  if (c != "`" && c != "~") { return "" }
  n = 0
  while (substr(s, i + n, 1) == c) { n++ }
  if (n < 3) { return "" }
  if (c == "`" && index(substr(s, i + n), "`") > 0) { return "" }
  return c " " n
}

function fence_close(s, fc, fl,   code, n, c, i) {
  code = indent_width(s)
  if (int(code / 1000) > 3) { return 0 }
  i = code % 1000
  c = substr(s, i, 1)
  if (c != fc) { return 0 }
  n = 0
  while (substr(s, i + n, 1) == c) { n++ }
  if (n < fl) { return 0 }
  return is_ws(substr(s, i + n))
}

function emit(s) {
  buffer[++buffered] = s
}


BEGIN {
  in_comment = 0; fence_c = ""; fence_l = 0
  seen_visible = 0; removed_leading = 0; pending = 0; total = 0
}
{
  total++
  line = $0
  if (fence_c != "") {
    emit(line)
    seen_visible = 1
    if (fence_close(line, fence_c, fence_l)) { fence_c = ""; fence_l = 0 }
    next
  }
  f = fence_open(line)
  if (f != "") {
    split(f, part, " ")
    fence_c = part[1]
    fence_l = part[2] + 0
    emit(line)
    seen_visible = 1
    next
  }
  out = ""; i = 1; n = length(line); had = 0
  while (i <= n) {
    if (in_comment) {
      k = index(substr(line, i), "-->")
      if (k == 0) { had = 1; i = n + 1; break }
      i = i + k - 1 + 3
      in_comment = 0
      had = 1
      continue
    }
    c = substr(line, i, 1)
    if (c == "<" && substr(line, i, 4) == "<!--") {
      in_comment = 1; had = 1; i += 4; continue
    }
    if (c == "-" && substr(line, i, 3) == "-->") {
      print "release_notes_body: stray --> outside a literal-code context on line " NR > "/dev/stderr"
      failed = 1; failcode = 4
      exit 4
    }
    if (c == "`") {
      run = 0
      while (substr(line, i + run, 1) == "`") { run++ }
      j = i + run; found = 0
      while (j <= n) {
        if (substr(line, j, 1) == "`") {
          m = 0
          while (substr(line, j + m, 1) == "`") { m++ }
          if (m == run) { found = 1; break }
          j += m
        } else { j++ }
      }
      if (found) {
        out = out substr(line, i, j + run - i)
        i = j + run
        continue
      }
      out = out substr(line, i, run)
      i += run
      continue
    }
    out = out c
    i++
  }
  if (had && !seen_visible) { removed_leading = 1 }
  # A line whose only content was a comment disappears entirely; turning it into a
  # blank line would leave the assembled body longer than source-minus-comment.
  if (had && is_ws(out)) { next }
  if (!seen_visible) {
    if (is_ws(out)) { pending++; next }
    if (removed_leading) { pending = 0 }
    while (pending > 0) { emit(""); pending-- }
  }
  seen_visible = 1
  emit(out)
}
END {
  if (in_comment) {
    print "release_notes_body: unclosed HTML comment at end of file" > "/dev/stderr"
    failed = 1
    exit 3
  }
  if (total == 0) {
    print "release_notes_body: input is empty" > "/dev/stderr"
    exit 5
  }
  if (!seen_visible) {
    print "release_notes_body: sanitised body is empty" > "/dev/stderr"
    exit 5
  }
  # Emit only now: a failure above must leave stdout completely empty so a caller
  # that ignores the exit status still has nothing usable to publish. awk still
  # runs END after exit, so the failure flag is what keeps this quiet.
  if (failed) { exit failcode }
  for (k = 1; k <= buffered; k++) {
    print buffer[k]
  }
}
AWK
}

release_notes_body() {
  # Every name this function touches is local, so a caller keeps its own values
  # for names like notes, program or cr_bytes.
  local notes="${1-}"
  local program
  local cr_bytes
  local cr_digits

  if [ -z "$notes" ] || [ ! -e "$notes" ]; then
    printf 'release_notes_body: notes file is missing: %s\n' "${notes:-<empty>}" >&2
    return 2
  fi
  if [ ! -f "$notes" ] || [ ! -r "$notes" ]; then
    printf 'release_notes_body: notes file is not a readable regular file: %s\n' "$notes" >&2
    return 2
  fi
  if [ ! -s "$notes" ]; then
    printf 'release_notes_body: notes file is empty: %s\n' "$notes" >&2
    return 5
  fi

  # The input must be LF-only. Counting CR bytes with a byte filter is exact and
  # platform-independent: grep and awk may translate line endings, tr does not.
  # A CR is refused instead of reinterpreted, so no byte is ever silently dropped.
  #
  # pipefail is enabled only inside the substitution, so a failing byte-counting
  # tool is reported instead of being masked by wc's exit status, and the whole
  # pipeline runs in a subshell so the caller's shell options stay untouched.
  # The `if !` form also keeps a caller that runs set -e from firing on it.
  if ! cr_bytes=$(set -o pipefail; LC_ALL=C tr -cd '\r' < "$notes" | wc -c); then
    printf 'release_notes_body: could not count carriage returns in %s\n' "$notes" >&2
    return 7
  fi
  # Accept only what a byte counter is allowed to print: one decimal number with
  # optional surrounding whitespace. Everything else is refused rather than
  # filtered, because deleting non-digits would launder '0garbage' into a count
  # of zero and silently disable the guard. Parameter expansion and case are used
  # deliberately -- no regex, so nothing here can touch a caller's BASH_REMATCH.
  cr_digits=${cr_bytes#"${cr_bytes%%[![:space:]]*}"}
  cr_digits=${cr_digits%"${cr_digits##*[![:space:]]}"}
  case "$cr_digits" in
    '' | *[!0-9]*)
      printf 'release_notes_body: carriage-return count was not a single number: %s\n' \
        "$notes" >&2
      return 7
      ;;
  esac
  # A non-zero decimal is any digit other than 0; testing for that avoids integer
  # range limits entirely.
  case "$cr_digits" in
    *[1-9]*)
      printf 'release_notes_body: notes file contains %s carriage return(s); this body must be LF-only: %s\n' \
        "$cr_digits" "$notes" >&2
      return 6
      ;;
  esac
  program="$(_release_notes_body_awk)"
  awk "$program" < "$notes"
}

if [ "${BASH_SOURCE[0]:-}" = "${0:-}" ]; then
  # Strict mode applies only to direct execution; a sourcing caller keeps its own
  # shell options untouched.
  set -u
  release_notes_body ${1+"$1"}
  exit $?
fi
