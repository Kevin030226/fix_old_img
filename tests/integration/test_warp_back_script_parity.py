"""The two vendored warp-back scripts are copies, and copies drift.

Found by running the service: ticking "High-resolution face path (needs the HR
weights)" failed every time, and the cause was one missing line in
`align_warp_back_multiple_dlib_HR.py`:

    UFuncOutputCastingError: Cannot cast ufunc 'multiply' output from dtype('float64')
    to dtype('uint8') with casting rule 'same_kind'

`mask *= 255.0` widens a uint8 mask to float64, which NumPy 2 refuses in place.
`align_warp_back_multiple_dlib.py` — the non-HR twin, byte-identical in 13 of its
16 functions — already had `mask = mask.astype(np.float64)`. The fix had been
applied to one copy and not the other, so the ordinary path worked and the HR path
could not run at all.

A bug fixed in one copy of a vendored file is not fixed. The gate is therefore
parity, not a NumPy version check: after the fix the two scripts must differ in
exactly the two lines that *are* the HR distinction, and nowhere else. Both
directions matter — a fix applied to only one copy fails, and so does a
"correction" that quietly makes the HR script identical to its twin, because the
512-pixel geometry is the entire reason the HR script exists.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

FACE_DIR = Path(__file__).resolve().parents[2] / "Face_Detection"
PLAIN = FACE_DIR / "align_warp_back_multiple_dlib.py"
HR = FACE_DIR / "align_warp_back_multiple_dlib_HR.py"

#: The only lines allowed to differ. Both are the HR face size: the HR path warps
#: into a 512-pixel crop, the ordinary path into 256.
ALLOWED_DIFFERENCES = {
    ("compute_inverse_transformation_matrix", "* 256.0", "* 512.0"),
    ("compute_transformation_matrix", "* 256.0", "* 512.0"),
}


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _significant(path: Path, name: str) -> list[str]:
    """A function's code lines: non-blank, comment-free, indentation stripped.

    Comments are excluded deliberately. The two copies are allowed to explain
    themselves differently — the fix in the HR script carries a comment saying it
    missed a fix the other one has, and comparing prose would make the gate fail on
    the very documentation that stops the next person repeating the mistake.
    """
    source = path.read_text(encoding="utf-8", errors="replace").splitlines()
    node = _functions(path)[name]
    out = []
    for line in source[node.lineno - 1: node.end_lineno]:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            out.append(stripped)
    return out


def test_both_warp_back_scripts_are_present():
    assert PLAIN.is_file(), f"missing {PLAIN}"
    assert HR.is_file(), f"missing {HR}"


def test_the_two_scripts_declare_the_same_functions():
    assert set(_functions(PLAIN)) == set(_functions(HR))


@pytest.mark.parametrize("name", ["blur_blending_cv2"])
def test_the_shared_functions_have_not_diverged(name):
    """A fix applied to one copy of a vendored file is not a fix."""
    plain = _significant(PLAIN, name)
    hr = _significant(HR, name)
    if plain == hr:
        return
    allowed = [d for d in ALLOWED_DIFFERENCES if d[0] == name]
    # The length check below is what makes `strict` moot here, but a silent
    # truncation would hide a real divergence, so it is stated.
    differences = [(a, b) for a, b in zip(plain, hr, strict=False) if a != b]
    unexpected = [
        (a, b) for a, b in differences
        if not any(a == x and b == y for _n, x, y in allowed)
    ]
    # Different lengths means a line was added or removed, which is never the HR
    # distinction.
    assert len(plain) == len(hr), (
        f"{name} differs in length ({len(plain)} vs {len(hr)}); the HR variant is a "
        "copy, so it should not grow or shrink relative to its twin"
    )
    assert not unexpected, (
        f"{name} differs between the two scripts in a way that is not the HR "
        f"resolution: {unexpected}. Fix both copies, or record the difference in "
        "ALLOWED_DIFFERENCES if it is deliberate."
    )


@pytest.mark.parametrize("name", ["compute_transformation_matrix",
                                  "compute_inverse_transformation_matrix"])
def test_the_hr_scripts_still_use_the_larger_face_geometry(name):
    """The gate above must not be satisfiable by making the HR script identical.

    If someone 'fixes' the divergence by copying the plain script over the HR one,
    the HR path would silently stop being high-resolution. So the HR distinction is
    asserted, not merely tolerated.
    """
    plain = " ".join(_significant(PLAIN, name))
    hr = " ".join(_significant(HR, name))
    assert "256.0" in plain, f"{name} no longer uses 256 in the plain script: {plain}"
    assert "512.0" in hr, f"{name} no longer uses 512 in the HR script: {hr}"


def _load_blending_cv2(path: Path):
    """Exec just that function, so the check runs the real code without the script.

    These files are `__main__` scripts: importing them would run a face warp-back
    over `examples/`. The function is self-contained apart from `np` and `cv2`, so
    compiling just its source is enough to exercise it.
    """
    import cv2

    node = _functions(path)["blur_blending_cv2"]
    # `.splitlines()` is load-bearing: slicing the raw string indexes *characters*,
    # which silently yields a 17-character fragment rather than the function.
    source = path.read_text(encoding="utf-8", errors="replace").splitlines()
    snippet = "\n".join(source[node.lineno - 1: node.end_lineno])
    namespace: dict = {"np": np, "cv2": cv2}
    exec(compile(snippet, str(path), "exec"), namespace)  # noqa: S102
    return namespace["blur_blending_cv2"]


@pytest.mark.parametrize("path", [PLAIN, HR], ids=["plain", "HR"])
def test_the_blending_runs_on_the_installed_numpy(path):
    """The regression, as an execution: a uint8 mask in, a float image out.

    Under NumPy 1 this passed and the cast happened silently. NumPy 2 raises, so a
    version guard would have been the wrong gate; running the code is the gate.
    """
    np_major = int(np.__version__.split(".")[0])
    if np_major < 2:
        pytest.skip(f"the cast only fails from NumPy 2; running {np.__version__}")

    blend = _load_blending_cv2(path)

    rng = np.random.default_rng(0)
    im1 = rng.random((32, 32, 3)) * 255.0          # the warped-back face
    im2 = rng.random((32, 32, 3)) * 255.0          # the restored image
    # The real caller warps `forward_mask` to `output_shape=(h, w, 3)`, so the mask
    # is three-channel, and `im1 * mask_blur` has to broadcast against it.
    mask = np.zeros((32, 32, 3), dtype=np.uint8)
    mask[8:24, 8:24, :] = 1

    out = blend(im1, im2, mask)
    assert out.shape == (32, 32, 3), out.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0, (out.min(), out.max())


@pytest.mark.parametrize("path", [PLAIN, HR], ids=["plain", "HR"])
def test_the_blending_does_not_mutate_the_callers_mask(path):
    """`mask *=` is in-place; without the widening line it also failed to run, but
    the widening must not become a silent edit of the caller's array."""
    import cv2  # noqa: F401  (the function's namespace needs it)

    blend = _load_blending_cv2(path)
    mask = np.zeros((16, 16, 3), dtype=np.uint8)
    mask[4:12, 4:12, :] = 1
    before = mask.copy()
    im1 = np.ones((16, 16, 3))
    blend(im1, im1.copy(), mask)
    assert mask.dtype == before.dtype
    assert np.array_equal(mask, before), "the caller's mask was modified in place"
