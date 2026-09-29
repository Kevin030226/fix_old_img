"""Every example image must be offered, and every offered path must exist.

Asked for directly: the example set should be complete. Two failure modes, in
opposite directions, and a one-directional test catches only one of them:

* an image sitting in `examples/` that no tab offers — invisible to a new arrival,
  who has no other way to know what the analyser will do with a photo they
  recognise;
* a tab pointing at a path that is not there — the thumbnail renders broken, and
  the Auto tab is the one place a user picks their first test.

The Auto tab is where the set was thin: three images, in V2 as well as V3, so
nothing had been lost by the refactoring. It is the tab whose plan is derived from
the picture, so it is the one that should offer everything the other tabs accept —
clean, scratched and grey.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = ROOT / "examples"
GRADIO_APP = ROOT / "src" / "fiximg" / "ui" / "gradio_app.py"


def _on_disk() -> set[str]:
    return {
        p.relative_to(ROOT).as_posix()
        for p in EXAMPLES_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }


def _declared_in_source() -> set[str]:
    """Every `./examples/...` path named in the UI, however it is spelled.

    The Auto tab builds its list from a comprehension over the letters, so a plain
    string search would find only the template. The letters are therefore expanded
    from the source rather than hard-coded here: a gate that repeats the list it is
    checking cannot notice the list changing.
    """
    source = GRADIO_APP.read_text(encoding="utf-8")
    found: set[str] = set()

    # Literal paths, e.g. ["./examples/old/a.png"]. Templates containing `{` are
    # skipped: they are a shape, not a path, and letting one through would make the
    # existence test look for a file literally named `{name}.png`.
    for match in re.finditer(r'"\./examples/[^"]+"', source):
        path = match.group(0).strip('"').lstrip("./")
        if "{" not in path:
            found.add(path)

    # Comprehension form: {name} for name in "abcdefgh", and for name in (...),
    # where the directory is spelled in the same f-string.
    pattern = re.compile(
        r'f"\./examples/(?P<dir>[a-z_]+)/\{name\}\.(?P<ext>png|jpg|jpeg)"\]\s*'
        r'for\s+name\s+in\s+(?P<items>"[^"]*"|\([^)]*\))'
    )
    for match in pattern.finditer(source):
        items = match.group("items")
        if items.startswith('"'):
            names = list(items.strip('"'))          # "abcdefgh" -> one name per letter
        else:
            names = re.findall(r'"([^"]*)"', items)  # ("o1", "o2") -> two names
        for name in names:
            found.add(f"examples/{match.group('dir')}/{name}.{match.group('ext')}")

    return found


def test_the_examples_directory_exists_and_is_not_empty():
    on_disk = _on_disk()
    assert on_disk, f"no example images under {EXAMPLES_DIR}"
    assert len(on_disk) >= 14, f"expected the full set, found {len(on_disk)}: {sorted(on_disk)}"


def test_every_example_image_on_disk_is_offered_in_the_ui():
    on_disk = _on_disk()
    declared = _declared_in_source()
    missing = sorted(on_disk - declared)
    assert not missing, (
        "these images are in the repository but no tab offers them, so they are "
        f"invisible to anyone who has not read the source: {missing}"
    )


def test_every_path_the_ui_offers_actually_exists():
    declared = _declared_in_source()
    missing = sorted(p for p in declared if not (ROOT / p).is_file())
    assert not missing, f"the UI points at example images that are not there: {missing}"


def _auto_tab_paths() -> set[str]:
    """The Auto tab's examples, expanded — the first `gr.Examples` in the file."""
    source = GRADIO_APP.read_text(encoding="utf-8")
    start = source.index("gr.Examples")
    end = source.index("submit_button_auto.click", start)
    return _expand_block(source[start:end])


def _expand_block(block: str) -> set[str]:
    """Every example path a source block offers, comprehensions included."""
    found: set[str] = set()
    for match in re.finditer(r'"\./examples/[^"]+"', block):
        path = match.group(0).strip('"').lstrip("./")
        if "{" not in path:
            found.add(path)
    pattern = re.compile(
        r'f"\./examples/(?P<dir>[a-z_]+)/\{name\}\.(?P<ext>png|jpg|jpeg)"\]\s*'
        r'for\s+name\s+in\s+(?P<items>"[^"]*"|\([^)]*\))'
    )
    for match in pattern.finditer(block):
        items = match.group("items")
        names = (list(items.strip('"')) if items.startswith('"')
                 else re.findall(r'"([^"]*)"', items))
        for name in names:
            found.add(f"examples/{match.group('dir')}/{name}.{match.group('ext')}")
    return found


def test_the_auto_tab_offers_the_whole_repository():
    """The tab whose plan is derived from the picture offers everything.

    Narrower on purpose than the coverage test above: it pins the Auto tab
    specifically, so a later edit that trims it again is caught here rather than by
    the general test, which would still pass while those three remained offered on
    the dedicated tabs.

    The comparison is on the *expanded* set, not on the source text — checking that
    the template string is still present would pass with one letter instead of eight,
    which is the whole failure being guarded against.
    """
    auto = _auto_tab_paths()
    on_disk = _on_disk()
    missing = sorted(on_disk - auto)
    assert not missing, (
        "the Auto tab no longer offers every example image: "
        f"{missing} (it offers {len(auto)} of {len(on_disk)})"
    )
    assert len(auto) == len(on_disk) == 14, (
        f"the Auto tab offers {len(auto)} and the repository holds {len(on_disk)}; "
        f"expected all 14: {sorted(auto)}"
    )


def test_the_paging_hint_is_present_wherever_the_list_is_long():
    """Gradio paginates a long example list; without the note the rest looks absent.

    This is the same note the Restore tab carries, and it is easy to drop when the
    list is edited — which would make a complete set look truncated.
    """
    source = GRADIO_APP.read_text(encoding="utf-8")
    for name, marker in (("Restore", "old/b.png"), ("Auto", "examples/color")):
        assert marker in source, f"the {name} tab's example list is gone"
    assert source.count("samples are on a second page") >= 2, (
        "both the Restore tab (8) and the Auto tab (14) paginate, so both need the "
        "note that tells a user to expand 'Pages'"
    )


@pytest.mark.parametrize("subdir,expected", [
    ("old", 8), ("old_w_scratch", 4), ("color", 2),
])
def test_each_group_is_intact(subdir, expected):
    files = sorted((EXAMPLES_DIR / subdir).glob("*"))
    files = [f for f in files if f.is_file()]
    assert len(files) == expected, (
        f"examples/{subdir} has {len(files)} files, expected {expected}: "
        f"{[f.name for f in files]}"
    )
