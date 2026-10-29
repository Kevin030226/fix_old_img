"""Behavioural tests for scripts/ci/release_notes_body.sh.

The release workflow composes the published Release body by copying the notes
file from docs/releases/v<version>.md verbatim. The notes file is allowed to
carry maintainer HTML comments, but they must never reach the public body: they
are invisible on the rendered page yet fully readable in the raw Markdown and
the API, so a stray authoring note becomes a permanent public artefact.

These tests judge the sanitiser by behaviour rather than by grepping for a
message: comments disappear, everything else -- fenced code, inline code spans,
visible HTML, bare angle-bracket text, line endings -- survives byte for byte,
and malformed input fails closed instead of publishing a mangled body.
"""
import os
import shutil
import subprocess

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HELPER = os.path.join(PROJECT_ROOT, "scripts", "ci", "release_notes_body.sh").replace(os.sep, "/")
RELEASES = os.path.join(PROJECT_ROOT, "docs", "releases")


def _find_bash():
    """Return a bash that can reach this checkout, or None.

    On Windows a bare ``bash`` is usually the WSL stub in system32, whose view of
    drive-letter paths is a different filesystem, so Git-for-Windows candidates are
    tried first and every candidate must prove it can see a real file here.
    """
    candidates = []
    if os.name == "nt":
        for env in ("ProgramFiles", "ProgramFiles(x86)", "LocalAppData"):
            base = os.environ.get(env)
            if not base:
                continue
            candidates.append(os.path.join(base, "Git", "bin", "bash.exe"))
            candidates.append(os.path.join(base, "Programs", "Git", "bin", "bash.exe"))
    found = shutil.which("bash") or shutil.which("bash.exe")
    if found and "Windows" not in found:
        candidates.append(found)

    probe_path = PROJECT_ROOT.replace(os.sep, "/")
    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        try:
            probe = subprocess.run(
                [candidate, "-c", 'test -f "$1/pyproject.toml" && git -C "$1" rev-parse --git-dir',
                 "_", probe_path],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0:
            return candidate
    return None


BASH = _find_bash()

pytestmark = pytest.mark.skipif(
    BASH is None or not os.path.exists(HELPER),
    reason="no bash able to run the helper on this host",
)


def _run(path, cwd=None):
    """Run the helper on ``path`` and return the completed process as text."""
    return subprocess.run(
        [BASH, HELPER, str(path)],
        cwd=cwd or PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )


def _run_bytes(path, cwd=None):
    """Run the helper and return raw bytes, so CRLF is not folded by the reader."""
    return subprocess.run(
        [BASH, HELPER, str(path)],
        cwd=cwd or PROJECT_ROOT,
        capture_output=True,
        timeout=180,
    )


def _write(tmp_path, text, name="notes.md", newline=None):
    """Write notes text, optionally with explicit CRLF endings."""
    path = tmp_path / name
    if newline is None:
        path.write_text(text, encoding="utf-8", newline="")
    else:
        path.write_text(text, encoding="utf-8", newline=newline)
    return path



# --- content preservation ------------------------------------------------


def test_comment_free_input_is_byte_identical(tmp_path):
    source = "# Title\n\nBody text.\n\n## Section\n\nMore text.\n"
    result = _run(_write(tmp_path, source))
    assert result.returncode == 0, result.stderr
    assert result.stdout == source


def test_leading_single_line_comment_is_removed(tmp_path):
    source = "<!-- maintainer note -->\n# Title\n\nBody.\n"
    result = _run(_write(tmp_path, source))
    assert result.returncode == 0, result.stderr
    assert result.stdout == "# Title\n\nBody.\n"


def test_leading_multiline_comment_is_removed_with_its_leading_blanks(tmp_path):
    source = "<!--\nfive\nline\nmaintainer\nnote\n-->\n\n# Title\n\nBody.\n"
    result = _run(_write(tmp_path, source))
    assert result.returncode == 0, result.stderr
    assert result.stdout == "# Title\n\nBody.\n"


def test_multiple_comments_are_all_removed(tmp_path):
    source = "<!-- one -->\n# Title\n\n<!-- two\nspans lines -->\n\nBody.\n"
    result = _run(_write(tmp_path, source))
    assert result.returncode == 0, result.stderr
    # The blank lines that surrounded the mid-file comment survive; only the
    # comment's own lines disappear.
    assert result.stdout == "# Title\n\n\nBody.\n"


def test_visible_prefix_and_suffix_on_the_same_line_are_kept(tmp_path):
    source = "keep before <!-- drop me --> keep after\n"
    result = _run(_write(tmp_path, source))
    assert result.returncode == 0, result.stderr
    assert result.stdout == "keep before  keep after\n"


def test_backtick_fence_preserves_comment_delimiters(tmp_path):
    source = "```html\n<!-- this is sample markup -->\n```\n\nAfter.\n"
    result = _run(_write(tmp_path, source))
    assert result.returncode == 0, result.stderr
    assert result.stdout == source


def test_tilde_fence_preserves_comment_delimiters(tmp_path):
    source = "~~~\n<!-- keep -->\n-->\n~~~\n"
    result = _run(_write(tmp_path, source))
    assert result.returncode == 0, result.stderr
    assert result.stdout == source


def test_longer_opening_fence_is_closed_only_by_a_long_enough_run(tmp_path):
    source = "````\n```\n<!-- still code -->\n````\n\nAfter.\n"
    result = _run(_write(tmp_path, source))
    assert result.returncode == 0, result.stderr
    assert result.stdout == source


def test_inline_code_span_preserves_comment_delimiters(tmp_path):
    source = "Use `<!--` and `-->` verbatim, then <!-- drop --> done.\n"
    result = _run(_write(tmp_path, source))
    assert result.returncode == 0, result.stderr
    assert result.stdout == "Use `<!--` and `-->` verbatim, then  done.\n"


def test_visible_html_is_preserved(tmp_path):
    source = "<details>\n<summary>More</summary>\n\nHidden body.\n</details>\n"
    result = _run(_write(tmp_path, source))
    assert result.returncode == 0, result.stderr
    assert result.stdout == source


def test_literal_angle_bracket_text_is_preserved(tmp_path):
    source = "Fetches into `refs/release-tags/<tag>` for `v<pyproject version>.md`.\n"
    result = _run(_write(tmp_path, source))
    assert result.returncode == 0, result.stderr
    assert result.stdout == source


# --- line-ending contract: LF only, CR refused ----------------------------


def test_uniform_crlf_input_is_refused(tmp_path):
    """A CRLF file is rejected, not silently rewritten to LF or to CRLF."""
    source = "<!-- note -->\r\n# Title\r\n\r\nBody.\r\n"
    result = _run_bytes(_write(tmp_path, source, newline=""))
    assert result.returncode == 6
    assert result.stdout == b""
    assert b"carriage return" in result.stderr


def test_standalone_carriage_return_inside_visible_text_is_refused(tmp_path):
    """A CR that is content must not be deleted as if it were a line ending."""
    result = _run(_write(tmp_path, "# Title\n\nA\rB\n"))
    assert result.returncode == 6
    assert result.stdout == ""


def test_mixed_lf_and_crlf_is_refused(tmp_path):
    """Mixed endings are ambiguous, so the helper refuses instead of guessing."""
    result = _run(_write(tmp_path, "# Title\n\nBody.\r\n"))
    assert result.returncode == 6
    assert result.stdout == ""


def test_checked_in_release_notes_are_lf_only():
    """The premise of the LF-only contract: the notes we ship carry no CR."""
    for name in ("v3.1.1.md", "v3.1.2.md"):
        path = os.path.join(RELEASES, name)
        if not os.path.exists(path):
            pytest.skip(f"{name} is not present in this checkout")
        with open(path, "rb") as handle:
            data = handle.read()
        assert b"\r" not in data, f"{name} contains a carriage return"


# --- final-newline contract: canonical single terminator -------------------


def test_input_without_a_final_newline_gains_exactly_one(tmp_path):
    """Documented: the body is canonical Markdown, so it ends with one LF."""
    result = _run_bytes(_write(tmp_path, "# Title\n\nBody."))
    assert result.returncode == 0
    assert result.stdout == b"# Title\n\nBody.\n"


def test_input_with_a_final_newline_keeps_exactly_one(tmp_path):
    result = _run_bytes(_write(tmp_path, "# Title\n\nBody.\n"))
    assert result.returncode == 0
    assert result.stdout == b"# Title\n\nBody.\n"
    assert not result.stdout.endswith(b"\n\n")



# --- fail-closed behaviour -----------------------------------------------


def test_missing_file_fails_closed(tmp_path):
    missing = tmp_path / "absent.md"
    result = _run(missing)
    assert result.returncode == 2
    assert "missing" in result.stderr
    assert result.stdout == ""


def test_empty_file_fails_closed(tmp_path):
    result = _run(_write(tmp_path, ""))
    assert result.returncode == 5
    assert result.stdout == ""


def test_whitespace_only_file_fails_closed(tmp_path):
    result = _run(_write(tmp_path, "   \n\t\n\n"))
    assert result.returncode == 5
    assert result.stdout == ""


def test_unclosed_comment_fails_closed(tmp_path):
    source = "# Title\n\n<!-- never closed\n\nmore text\n"
    result = _run(_write(tmp_path, source))
    assert result.returncode == 3
    assert "unclosed" in result.stderr


def test_stray_closing_delimiter_fails_closed(tmp_path):
    source = "# Title\n\nstray --> here\n"
    result = _run(_write(tmp_path, source))
    assert result.returncode == 4
    assert "stray" in result.stderr


def test_comment_only_file_fails_closed(tmp_path):
    source = "<!--\nonly a comment\n-->\n"
    result = _run(_write(tmp_path, source))
    assert result.returncode == 5
    assert result.stdout == ""


def test_diagnostics_never_pollute_stdout(tmp_path):
    result = _run(_write(tmp_path, "# Title\n\nstray --> here\n"))
    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr.strip() != ""


# --- hostile filenames and contents ---------------------------------------


def test_filename_with_spaces_and_metacharacters(tmp_path):
    odd = tmp_path / "release notes $(touch x) `id` ; & .md"
    odd.write_text("# Title\n\n<!-- drop -->\nBody.\n", encoding="utf-8", newline="")
    result = _run(odd, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "# Title\n\nBody.\n"
    assert not (tmp_path / "x").exists()


def test_injection_shaped_content_is_never_executed(tmp_path):
    marker = tmp_path / "pwned.txt"
    source = (
        "# Title\n\n"
        "$(touch " + marker.name + ")\n"
        "`touch " + marker.name + "`\n"
        "<!-- $(touch " + marker.name + ") -->\n"
    )
    result = _run(_write(tmp_path, source), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert not marker.exists(), "note contents were executed"
    assert "$(touch pwned.txt)" in result.stdout



# --- the real release notes ------------------------------------------------


def _real_notes(name):
    """Return the notes text, or None when the file is not in this checkout."""
    path = os.path.join(RELEASES, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", newline="") as handle:
        return handle.read()


def _expected_without_top_level_comment(text):
    """Drop the single line-delimited <!-- ... --> block these notes carry."""
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line.strip() == "<!--")
    end = next(i for i, line in enumerate(lines) if line.strip() == "-->" and i > start)
    return "\n".join(lines[:start] + lines[end + 1:])


@pytest.mark.parametrize("name", ["v3.1.1.md", "v3.1.2.md"])
def test_real_release_notes_sanitise_to_the_expected_body(tmp_path, name):
    text = _real_notes(name)
    if text is None:
        pytest.skip(f"{name} is not present in this checkout")
    assert "<!--" in text, "these notes are expected to carry a maintainer comment"
    result = _run(os.path.join(RELEASES, name))
    assert result.returncode == 0, result.stderr
    assert "<!--" not in result.stdout
    assert "-->" not in result.stdout
    assert result.stdout == _expected_without_top_level_comment(text)


def test_real_notes_keep_their_inline_code_and_headings():
    text = _real_notes("v3.1.2.md")
    if text is None:
        pytest.skip("v3.1.2 notes are not present in this checkout")
    result = _run(os.path.join(RELEASES, "v3.1.2.md"))
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("# fiximg v3.1.2")
    assert "## Summary" in result.stdout
    assert "`refs/release-tags/<tag>`" in result.stdout


# --- documented contract ---------------------------------------------------


def test_only_outside_comment_is_removed_when_a_fence_holds_one(tmp_path):
    source = (
        "<!-- drop this -->\n"
        "```\n"
        "<!-- keep this -->\n"
        "```\n"
        "\n"
        "Body.\n"
    )
    result = _run(_write(tmp_path, source))
    assert result.returncode == 0, result.stderr
    assert result.stdout == "```\n<!-- keep this -->\n```\n\nBody.\n"


def test_blank_lines_between_sections_are_preserved_exactly(tmp_path):
    source = "# Title\n\n\n## Section\n\nBody.\n"
    result = _run(_write(tmp_path, source))
    assert result.returncode == 0, result.stderr
    assert result.stdout == source


# --- sourceability ---------------------------------------------------------

#: Runs in a fresh bash: sources the helper, then reports the shell option state
#: and whether any caller-owned variable was overwritten.
_SOURCE_PROBE = r"""
set +u
notes=SENTINEL
program=SENTINEL
cr_bytes=SENTINEL
crlf=SENTINEL
. "$1"
case $- in *u*) printf 'source=on\n' ;; *) printf 'source=off\n' ;; esac
if release_notes_body "$2" > /dev/null 2>&1; then printf 'call=ok\n'; else printf 'call=failed\n'; fi
printf 'notes=%s\n' "$notes"
printf 'program=%s\n' "$program"
printf 'cr_bytes=%s\n' "$cr_bytes"
printf 'crlf=%s\n' "$crlf"
case $- in *u*) printf 'after=on\n' ;; *) printf 'after=off\n' ;; esac
"""


def _probe(notes_path):
    return subprocess.run(
        [BASH, "-c", _SOURCE_PROBE, "_", HELPER, str(notes_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )


def test_sourcing_the_helper_does_not_enable_nounset(tmp_path):
    """Sourcing a library must not change the caller's shell options."""
    result = _probe(_write(tmp_path, "# Title\n\nBody.\n"))
    assert result.returncode == 0, result.stderr
    lines = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert lines["source"] == "off", "sourcing the helper enabled nounset"
    assert lines["after"] == "off", "calling the function enabled nounset"


def test_calling_the_helper_does_not_clobber_caller_variables(tmp_path):
    result = _probe(_write(tmp_path, "# Title\n\nBody.\n"))
    assert result.returncode == 0, result.stderr
    lines = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert lines["call"] == "ok", result.stdout
    for name in ("notes", "program", "cr_bytes", "crlf"):
        assert lines[name] == "SENTINEL", f"{name} leaked out of the function"


def test_direct_execution_still_works(tmp_path):
    """The file is both a sourceable library and a runnable script."""
    notes = _write(tmp_path, "# Title\n\n<!-- drop -->\nBody.\n")
    result = _run(notes)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "# Title\n\nBody.\n"


def test_multiple_trailing_blank_lines_are_preserved_byte_for_byte(tmp_path):
    """The body is newline-terminated, but it never trims trailing blank lines."""
    source = "# Title\n\nBody.\n\n\n"
    result = _run_bytes(_write(tmp_path, source))
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert result.stdout == b"# Title\n\nBody.\n\n\n"
    assert len(result.stdout) - len(result.stdout.rstrip(b"\n")) == 3


def _run_with_broken_tool(name, notes_path):
    """Run the helper with ``name`` replaced by a failing shell function.

    Shadowing through an exported function, rather than a fake executable on PATH,
    keeps the failure deterministic on every platform: a file-based stub needs an
    executable bit, which the Windows POSIX layer does not reliably honour.
    """
    wrapper = (
        f"{name}() {{ return 3; }}\n"
        f"export -f {name}\n"
        'exec "$1" "$2"\n'
    )
    return subprocess.run(
        [BASH, "-c", wrapper, "_", HELPER, str(notes_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )


def test_a_failing_byte_counter_is_reported_not_masked(tmp_path):
    """Without a pipefail scope the last pipeline command would hide the failure."""
    notes = _write(tmp_path, "# Title\n\nBody.\n")
    result = _run_with_broken_tool("tr", notes)
    assert result.returncode == 7, result.stdout
    assert result.stdout == ""
    assert "carriage" in result.stderr


def test_a_failing_word_counter_is_also_reported(tmp_path):
    notes = _write(tmp_path, "# Title\n\nBody.\n")
    result = _run_with_broken_tool("wc", notes)
    assert result.returncode == 7, result.stdout
    assert result.stdout == ""


def test_tool_failure_leaves_caller_state_untouched(tmp_path):
    """The failing path must not leak shell options or variables either."""
    notes = _write(tmp_path, "# Title\n\nBody.\n")
    probe = (
        'set +u\n'
        'notes=SENTINEL\n'
        'program=SENTINEL\n'
        'cr_bytes=SENTINEL\n'
        'tr() { return 3; }\n'
        '. "$1"\n'
        'release_notes_body "$2" > /dev/null 2>&1\n'
        'printf "rc=%s\\n" "$?"\n'
        'printf "notes=%s\\n" "$notes"\n'
        'printf "program=%s\\n" "$program"\n'
        'printf "cr_bytes=%s\\n" "$cr_bytes"\n'
        'case $- in *u*) printf "option=on\\n" ;; *) printf "option=off\\n" ;; esac\n'
    )
    result = subprocess.run(
        [BASH, "-c", probe, "_", HELPER, str(notes)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    lines = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert lines["rc"] == "7", result.stdout
    assert lines["notes"] == "SENTINEL"
    assert lines["program"] == "SENTINEL"
    assert lines["cr_bytes"] == "SENTINEL"
    assert lines["option"] == "off"


def _run_with_counter(output, notes_path):
    """Run the helper with ``wc`` replaced by a function that prints ``output``.

    The stub consumes stdin so the upstream tr does not die of SIGPIPE, then
    exits successfully. That is exactly the shape that used to slip past the
    guard: a successful counter whose text was not a clean number.
    """
    wrapper = (
        'wc() { cat > /dev/null; printf "%s" "$P79_COUNTER_OUT"; }\n'
        'export -f wc\n'
        'exec "$1" "$2"\n'
    )
    env = dict(os.environ)
    env["P79_COUNTER_OUT"] = output
    return subprocess.run(
        [BASH, "-c", wrapper, "_", HELPER, str(notes_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=180,
    )


@pytest.mark.parametrize(
    "output",
    [
        "0garbage",
        "garbage0",
        "x1y",
        "+1",
        "-1",
        "12 34",
        "12\n34\n",
        "12.",
        "12,5",
        "0x10",
        "",
        "   ",
    ],
)
def test_malformed_but_successful_counter_output_is_refused(tmp_path, output):
    """A successful counter that is not one number must not become a count."""
    notes = _write(tmp_path, "# Title\n\nBody.\n")
    result = _run_with_counter(output, notes)
    assert result.returncode == 7, (output, result.stdout)
    assert result.stdout == ""
    assert result.stderr.strip() != ""


def test_whitespace_padded_zero_count_is_accepted(tmp_path):
    notes = _write(tmp_path, "# Title\n\nBody.\n")
    for output in (" 0", "0\n", "   0   \n"):
        result = _run_with_counter(output, notes)
        assert result.returncode == 0, (output, result.stderr)
        assert result.stdout == "# Title\n\nBody.\n"


def test_whitespace_padded_nonzero_count_is_treated_as_cr_present(tmp_path):
    notes = _write(tmp_path, "# Title\n\nBody.\n")
    for output in ("  12  ", "3\n"):
        result = _run_with_counter(output, notes)
        assert result.returncode == 6, (output, result.stdout)
        assert result.stdout == ""


def test_malformed_count_cannot_bypass_the_cr_guard(tmp_path):
    """The exact fail-open case: a lying counter over a CRLF note."""
    crlf = _write(tmp_path, "# Title\r\n\r\nBody.\r\n", newline="")
    result = _run_with_counter("0garbage", crlf)
    assert result.returncode == 7
    assert result.stdout == ""


def test_strict_parsing_leaves_caller_scratch_state_untouched(tmp_path):
    """Validation must not clobber BASH_REMATCH or the other sentinels."""
    notes = _write(tmp_path, "# Title\n\nBody.\n")
    probe = (
        'set +u\n'
        "[[ 'abc123' =~ ([a-z]+)([0-9]+) ]]\n"
        'printf "before=%s:%s\\n" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"\n'
        'notes=KEEP\n'
        'cr_bytes=KEEP\n'
        'cr_digits=KEEP\n'
        'program=KEEP\n'
        'wc() { cat > /dev/null; printf "0garbage"; }\n'
        '. "$1"\n'
        'release_notes_body "$2" > /dev/null 2>&1\n'
        'printf "rc=%s\\n" "$?"\n'
        'printf "after=%s:%s\\n" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"\n'
        'printf "notes=%s\\n" "$notes"\n'
        'printf "cr_bytes=%s\\n" "$cr_bytes"\n'
        'printf "cr_digits=%s\\n" "$cr_digits"\n'
        'printf "program=%s\\n" "$program"\n'
        'case $- in *u*) printf "option=on\\n" ;; *) printf "option=off\\n" ;; esac\n'
    )
    result = subprocess.run(
        [BASH, "-c", probe, "_", HELPER, str(notes)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    lines = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert lines["rc"] == "7", result.stdout
    assert lines["before"] == "abc:123"
    assert lines["after"] == "abc:123", "the parser clobbered BASH_REMATCH"
    assert lines["notes"] == "KEEP"
    assert lines["cr_bytes"] == "KEEP"
    assert lines["cr_digits"] == "KEEP"
    assert lines["program"] == "KEEP"
    assert lines["option"] == "off"
