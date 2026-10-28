"""Publication-safety gate (P5).

The P4 audit found the project was safe to publish *after* a small set of
fixes: local pollution directories that a `git add -A` would have committed,
machine paths inside tracked reports, and no explicit least-privilege block on
the CI workflow. Those were fixed by hand. This module is what keeps them
fixed -- every rule below failed at least once before it was written.

What is asserted here
---------------------
* the pollution directory names are ignored by both lists, and root-anchored so
  a legitimate directory of the same name deeper in the tree still ships;
* machine-specific paths and secrets do not reappear in publishable material;
* execution evidence (JUnit XML, raw pytest logs) is never a commit candidate;
* model weights and user data stay ignored;
* the CI workflow keeps an explicit least-privilege `permissions:` block while
  the release workflow keeps the write access publishing actually needs.

What is deliberately *not* flagged
----------------------------------
The allowlist below is not a convenience. A gate that cries wolf on the
project's own public identity, on the fixed test credential `fiximg:fiximg`,
or on the `REPLACE_WITH_*` placeholders in `config/users.example.yaml` would be
switched off within a week, and then it protects nothing. Each entry records
why it is safe, so a future reader can tell an exemption from an oversight.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GITIGNORE = ROOT / '.gitignore'
DOCKERIGNORE = ROOT / '.dockerignore'
REPORTS = ROOT / 'reports'
EVIDENCE = REPORTS / 'evidence'
WORKFLOWS = ROOT / '.github' / 'workflows'

#: Root-level names produced by an unset variable or an interrupted pip build.
#: They carry the local absolute paths they were built from, so they must never
#: be stageable and never enter a build context.
POLLUTION_NAMES = [
    '%SystemDrive%',
    'undefined',
    '~',
    'pip-build-env-58bjs33q',
    'pip-build-tracker-6vkogeg2',
    'pip-ephem-wheel-cache-amhsltf1',
    'pip-install-08zd6xye',
]

#: The glob forms the two ignore files use to cover those names generically.
POLLUTION_PATTERNS = [
    '%SystemDrive%',
    'undefined',
    '~',
    'pip-build-env-*',
    'pip-build-tracker-*',
    'pip-ephem-wheel-cache-*',
    'pip-install-*',
]

#: Paths that are local material, never publishable, in either list.
#:
#: Note what is *not* here: `Global/` and `Face_Enhancement/` are vendored
#: **source** (50 and 37 modules of the upstream restoration pipeline) and are
#: published with the project. Only their `checkpoints/` subdirectories hold
#: weights, and those are excluded by extension. Listing the parent directories
#: would have excluded the source along with the weights -- the same class of
#: defect as the unanchored `storage/` rule this suite already guards against.
NEVER_PUBLISHED = [
    'admin_data',
    'storage',
]

#: Weight *files* rather than weight directories. `Global/` and `Face_Enhancement/`
#: also hold the vendored source of the restoration pipeline, so excluding the
#: directories would delete 87 publishable modules along with the checkpoints.
#: These samples are the real weight locations in this tree.
NEVER_PUBLISHED_FILES = [
    'weights/ddcolor/pytorch_model.pt',
    'Global/checkpoints/restoration/VAE_A_quality/latest_net_G.pth',
    'Face_Enhancement/checkpoints/FaceSR_512/latest_net_G.pth',
    'Face_Detection/shape_predictor_68_face_landmarks.dat',
]


def _entries(path: Path) -> list[str]:
    """Pattern lines as written, comments and blanks removed."""
    out: list[str] = []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            out.append(line)
    return out


def _matches(rel: str, patterns: list[str]) -> bool:
    """gitignore semantics, limited to the shapes these lists use."""
    import fnmatch

    parts = rel.split('/')
    for pattern in patterns:
        candidate = pattern.rstrip('/')
        anchored = candidate.startswith('/')
        bare = candidate.lstrip('/')
        if '/' in bare or anchored:
            if fnmatch.fnmatch(rel, bare) or rel.startswith(bare + '/'):
                return True
        else:
            if any(fnmatch.fnmatch(part, bare) for part in parts):
                return True
    return False


def _report_files() -> list[Path]:
    """Every Markdown report at the top of reports/ (evidence/ excluded)."""
    if not REPORTS.is_dir():
        return []
    return sorted(p for p in REPORTS.glob('*.md') if p.is_file())


# --------------------------------------------------------------- allowlist

#: Literals that look like a finding but are provably safe. Every entry is a
#: (label, regex) pair; a match is dropped before the real patterns run.
ALLOWED = [
    # The project's public author. It is in pyproject.toml, the README badges
    # and release.yml on purpose -- it is how the project is attributed.
    ('public author handle', re.compile(r'Kevin030226')),
    # The fixed development credential the tests and the CI service use.
    ('test database credential', re.compile(r'fiximg:fiximg')),
    ('test storage credential', re.compile(r'fiximg-secret')),
    # Placeholders that announce themselves.
    ('explicit placeholder', re.compile(r'REPLACE_WITH_[A-Z_]+')),
    ('angle-bracket placeholder', re.compile(r'<[a-z][a-z0-9-]*>')),
    # A documented generic example path, not this machine's.
    ('generic example path', re.compile(r'[A-Za-z]:/(code|path|srv|work)/')),
    # An elided path: a drive, a well-known root and a literal ellipsis. It names
    # no account and no project; it is how the docs discuss path leakage.
    ('elided windows path', re.compile('[' + 'A-Za-z' + ']' + ':' + re.escape(chr(92)) + 'Users' + re.escape(chr(92)) + re.escape('...'))),
    ('elided posix path', re.compile(r'[A-Za-z]:/Users/\.\.\.')),
    # A well-known install location under a placeholder home. The placeholder
    # was already substituted by the rule above, so this matches the remainder.
    ('placeholder-home tool path', re.compile(r'<allowed>' + re.escape(chr(92)) + r'AppData' + re.escape(chr(92)))),
    ('unix generic path', re.compile(r'/(home|Users)/<[a-z-]+>')),
    # Test-only hosts and localhost DSNs.
    ('localhost dsn', re.compile(r'(postgresql|redis|mysql)://[^\s:@]+:[^\s:@]+@(localhost|127\.0\.0\.1|postgres|redis):')),
]


def _strip_allowed(text: str) -> str:
    r"""Remove allowlisted spans so they cannot trip the real patterns.

    Applied to a fixpoint: substituting one placeholder can expose a second
    allowlisted shape (`<user-home>\AppData\...` only becomes matchable once
    `<user-home>` has been replaced), and a single pass would leave it behind.
    """
    for _ in range(5):
        before = text
        for _label, pattern in ALLOWED:
            text = pattern.sub('<allowed>', text)
        if text == before:
            break
    return text


#: Patterns that must never appear in publishable material.
MACHINE_LEAK_PATTERNS = [
    ('windows user home', re.compile(r'[A-Za-z]:\\Users\\[A-Za-z0-9._-]{3,}')),
    ('windows user home (posix)', re.compile(r'[A-Za-z]:/Users/[A-Za-z0-9._-]{3,}')),
    ('real drive project path', re.compile(r'[A-Za-z]:[\\/]fix_old_img')),
    ('conda environment', re.compile(r'[A-Za-z]:[\\/]Soft[\\/]miniconda3')),
    ('docker desktop install', re.compile(r'AppData[\\/](Local|Roaming)[\\/]Programs[\\/]DockerDesktop')),
    ('windows temp', re.compile(r'[A-Za-z]:[\\/]Windows[\\/]Temp')),
    ('posix home with real user', re.compile(r'(?<![<\w])/home/[a-z][a-z0-9_-]{2,}/')),
    ('wsl mount', re.compile(r'/mnt/[cdef]/Users/')),
    ('machine hostname', re.compile(r'\bLAPTOP-[A-Z0-9]{6,}\b')),
    ('session id', re.compile(r'\bpytest-of-[A-Za-z0-9-]+\b')),
]

SECRET_PATTERNS = [
    ('private key header', re.compile(r'-----BEGIN (RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY')),
    ('aws access key', re.compile(r'\bAKIA[0-9A-Z]{16}\b')),
    ('github token', re.compile(r'\bgh[pousr]_[A-Za-z0-9]{36,}')),
    ('github fine-grained pat', re.compile(r'\bgithub_pat_[A-Za-z0-9_]{82}')),
    ('slack token', re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{10,}')),
    ('stripe key', re.compile(r'\b(sk|rk)_(live|test)_[A-Za-z0-9]{24,}')),
    ('openai key', re.compile(r'\bsk-[A-Za-z0-9]{20,}T3BlbkFJ')),
    ('huggingface token', re.compile(r'\bhf_[A-Za-z0-9]{34,}')),
    ('google api key', re.compile(r'\bAIza[0-9A-Za-z_-]{35}')),
    ('pypi token', re.compile(r'\bpypi-[A-Za-z0-9_-]{50,}')),
    ('jwt', re.compile(r'\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}')),
    ('s3 presigned', re.compile(r'X-Amz-Signature=[0-9a-f]{64}')),
    ('embedded password dsn', re.compile(r'(postgresql|mysql|mongodb|amqp)(\+\w+)?://[^\s:@/<]+:[^\s:@/<]+@')),
]


# --------------------------------------------------- ignore-rule coverage

@pytest.mark.parametrize('name', POLLUTION_PATTERNS)
def test_pollution_is_ignored_by_both_lists(name):
    """The P4 finding: these names were stageable and would have been committed."""
    git = _entries(GITIGNORE)
    docker = _entries(DOCKERIGNORE)
    missing_git = not _matches(name, git)
    missing_docker = not _matches(name, docker)
    assert not missing_git, f'{name} is not ignored by .gitignore and would be committed'
    assert not missing_docker, f'{name} is not ignored by .dockerignore and would enter the image'


@pytest.mark.parametrize('name', POLLUTION_PATTERNS)
def test_pollution_patterns_are_root_anchored(name):
    """Anchoring is what stops the rule from eating a legitimate source directory.

    An unanchored `undefined` or `~` matches that name at any depth. This
    repository already shipped that defect once with `storage/`, which excluded
    src/fiximg/infrastructure/storage/ from a commit while the files sat on
    disk. The pollution rules must not repeat it.
    """
    raw = _entries(GITIGNORE)
    matching = [e for e in raw if e.strip('/').lstrip('/') == name]
    assert matching, f'{name} has no .gitignore rule'
    for entry in matching:
        assert entry.startswith('/'), (
            f'.gitignore has {entry!r} unanchored; it would also match a directory'
            f' named {name!r} anywhere in the tree, including inside src/ or tests/'
        )


def test_legitimate_source_directories_are_not_ignored():
    """The negative control for the anchoring rule above.

    Every real package directory must survive the ignore lists. If a pollution
    pattern were written unanchored, one of these would disappear from the
    commit -- and the failure would only show up on a clean checkout.
    """
    git = _entries(GITIGNORE)
    legitimate = [
        'src/fiximg/infrastructure/storage/base.py',
        'src/fiximg/infrastructure/storage/s3.py',
        'src/fiximg/infrastructure/queue/redis.py',
        'tests/unit/test_publication_safety.py',
        'docker/compose.yaml',
        'docs/api.md',
        'scripts/export_locks.py',
        'app/admin_panel.py',
    ]
    offenders = [rel for rel in legitimate if _matches(rel, git)]
    assert not offenders, (
        f'these legitimate files are excluded by .gitignore: {offenders}. '
        'A pollution rule is matching too broadly.'
    )


@pytest.mark.parametrize('name', NEVER_PUBLISHED)
def test_user_data_is_never_publishable(name):
    """Uploads and the runtime database must stay out of both artifacts."""
    git = _entries(GITIGNORE)
    docker = _entries(DOCKERIGNORE)
    assert _matches(name, git), f'{name} is stageable'
    assert _matches(name, docker), f'{name} would reach the image'


@pytest.mark.parametrize('rel', NEVER_PUBLISHED_FILES)
def test_model_weights_are_never_publishable(rel):
    """The weight files themselves, checked by their real paths in this tree.

    Asserting on the directories would be wrong in both directions: `weights/`
    holds nothing else, but `Global/` and `Face_Enhancement/` are mostly source.
    """
    git = _entries(GITIGNORE)
    docker = _entries(DOCKERIGNORE)
    assert _matches(rel, git), f'{rel} is stageable'
    assert _matches(rel, docker), f'{rel} would reach the image'


def test_execution_evidence_is_never_a_commit_candidate():
    """JUnit XML and raw pytest logs carry the hostname and absolute paths."""
    git = _entries(GITIGNORE)
    for sample in ('reports/evidence/junit.xml', 'reports/evidence/pytest_full3.log'):
        assert _matches(sample, git), (
            f'{sample} is committable; execution evidence must stay local'
        )
    # The clean evidence files stay committable on purpose.
    for sample in ('reports/evidence/trivy_worker_result.txt',
                   'reports/evidence/modified_hashes.json'):
        assert not _matches(sample, git), (
            f'{sample} is excluded, but it contains no machine detail and is'
            ' published as scan evidence'
        )


def test_reports_policy_is_written_down():
    """The reports/ policy must be stated, not inferred.

    Three rules: evidence is never published, top-level reports are published
    only after redaction, and the audit reports must themselves be redacted.
    """
    text = GITIGNORE.read_text(encoding='utf-8')
    assert '/reports/evidence/*.xml' in text
    assert '/reports/evidence/*.log' in text
    assert 'reports' in text, 'the reports/ policy is not documented in .gitignore'


# ------------------------------------------------------- content scanning


@pytest.mark.parametrize('report', _report_files(), ids=lambda p: p.name)
def test_published_reports_carry_no_machine_detail(report):
    """Every top-level report is a publish candidate, so none may leak the machine.

    P4 found the project root and the Windows account directory inside reports
    that were not ignored. Redaction replaced them with <repo> and <user-home>;
    this keeps them replaced.
    """
    text = _strip_allowed(report.read_text(encoding='utf-8', errors='replace'))
    hits: list[str] = []
    for label, pattern in MACHINE_LEAK_PATTERNS:
        match = pattern.search(text)
        if match:
            hits.append(f'{label}: {match.group(0)[:24]}...')
    assert not hits, f'{report.name} leaks machine detail: {hits}'


@pytest.mark.parametrize('report', _report_files(), ids=lambda p: p.name)
def test_published_reports_carry_no_secret(report):
    """No real credential format may appear in a report."""
    text = _strip_allowed(report.read_text(encoding='utf-8', errors='replace'))
    hits = [label for label, pattern in SECRET_PATTERNS if pattern.search(text)]
    assert not hits, f'{report.name} contains secret-shaped material: {hits}'


def test_published_docs_carry_no_machine_detail():
    """Documentation is the most-read published surface, so it is checked too."""
    docs = ROOT / 'docs'
    if not docs.is_dir():
        pytest.skip('no docs directory')
    offenders: list[str] = []
    for path in sorted(docs.rglob('*.md')):
        text = _strip_allowed(path.read_text(encoding='utf-8', errors='replace'))
        for label, pattern in MACHINE_LEAK_PATTERNS:
            if pattern.search(text):
                offenders.append(f'{path.relative_to(ROOT).as_posix()}: {label}')
    assert not offenders, f'docs leak machine detail: {offenders}'


def test_publishable_source_carries_no_secret():
    """Source and scripts are published verbatim; scan them for real credentials."""
    targets: list[Path] = []
    for sub in ('src', 'scripts', 'app', 'docker'):
        base = ROOT / sub
        if base.is_dir():
            targets.extend(sorted(base.rglob('*.py')))
            targets.extend(sorted(base.rglob('*.yaml')))
            targets.extend(sorted(base.rglob('*.yml')))
    offenders: list[str] = []
    for path in targets:
        if '__pycache__' in path.parts:
            continue
        text = _strip_allowed(path.read_text(encoding='utf-8', errors='replace'))
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                offenders.append(f'{path.relative_to(ROOT).as_posix()}: {label}')
    assert not offenders, f'secret-shaped material in publishable source: {offenders}'


# --------------------------------------------------------- workflow permissions


def _workflow(name: str) -> dict:
    import yaml

    with open(WORKFLOWS / name, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def test_ci_declares_least_privilege_permissions():
    """P4 finding I-1: CI had no `permissions:` block and inherited the default.

    For older repositories that default is write-all, and a pull request from a
    fork then runs with a token that can push. CI only reads the checkout and
    pulls public images, so `contents: read` is the whole requirement.
    """
    document = _workflow('ci.yml')
    permissions = document.get('permissions')
    assert permissions is not None, (
        'ci.yml declares no permissions block, so it inherits the repository'
        ' default -- add `permissions: contents: read`'
    )
    assert permissions.get('contents') == 'read', (
        f'ci.yml grants {permissions!r}; CI needs only contents: read'
    )
    writes = {k: v for k, v in permissions.items() if v == 'write'}
    assert not writes, f'ci.yml grants write access it does not need: {writes}'


def test_no_ci_job_escalates_its_permissions():
    """A job-level block would silently override the workflow-level minimum."""
    document = _workflow('ci.yml')
    escalating: list[str] = []
    for job_id, job in (document.get('jobs') or {}).items():
        job_permissions = job.get('permissions')
        if not job_permissions:
            continue
        for scope, level in job_permissions.items():
            if level == 'write':
                escalating.append(f'{job_id}: {scope}: write')
    assert not escalating, f'CI jobs escalate permissions: {escalating}'


def test_release_keeps_the_write_access_publishing_requires():
    """The other half of the split: publishing a Release and pushing to GHCR.

    P39 narrowed the *workflow-level* block to read-only and moved write access
    onto the two jobs that actually publish. This asserts the capability is
    still there, and only there: dropping it would break releases, while
    spreading it to the read-only jobs would undo the least-privilege split.
    """
    document = _workflow('release.yml')
    permissions = document.get('permissions') or {}
    assert permissions.get('contents') == 'read', (
        'release.yml must declare contents: read at the top level; write access '
        'belongs to the publishing jobs only'
    )
    assert not {k: v for k, v in permissions.items() if v == 'write'}, (
        f'release.yml grants workflow-level write access: {permissions}'
    )

    jobs = document.get('jobs') or {}
    release_job = jobs.get('release') or {}
    assert (release_job.get('permissions') or {}).get('contents') == 'write', (
        'the release job lost contents: write; it cannot create a GitHub Release'
    )
    for job_id in ('stage-images', 'publish-version-tags', 'promote-latest'):
        job_permissions = (jobs.get(job_id) or {}).get('permissions') or {}
        assert job_permissions.get('packages') == 'write', (
            f'{job_id} lost packages: write; it cannot push to GHCR'
        )
        assert job_permissions.get('contents') == 'read', (
            f'{job_id} should keep contents: read; it only needs the checkout'
        )

    escalating = sorted(
        f'{job_id}:{scope}:{level}'
        for job_id, job in jobs.items()
        for scope, level in (job.get('permissions') or {}).items()
        if level == 'write'
        and job_id not in ('release', 'stage-images', 'publish-version-tags', 'promote-latest')
    )
    assert not escalating, f'non-publishing jobs escalate permissions: {escalating}'


def test_release_only_uses_the_github_token():
    """No third-party secret may be referenced by the publishing workflow."""
    text = (WORKFLOWS / 'release.yml').read_text(encoding='utf-8')
    referenced = set(re.findall(r'secrets\.([A-Za-z0-9_]+)', text))
    assert referenced <= {'GITHUB_TOKEN'}, (
        f'release.yml references secrets beyond GITHUB_TOKEN: {sorted(referenced)}'
    )


# ------------------------------------------------------ negative controls
#
# A gate that cannot fail is a comment. Each test below feeds a known-bad
# input to the same detector the real assertions use and requires it to fire.
# The samples are synthetic: none of them is a real secret or a real path
# from this machine.

BAD_MACHINE_SAMPLES = [
    ('windows home', 'C:' + chr(92) + 'Users' + chr(92) + 'someuser' + chr(92) + 'AppData'),
    ('drive project path', 'D:' + chr(92) + 'fix_old_img' + chr(92) + 'proj'),
    ('conda path', 'D:' + chr(92) + 'Soft' + chr(92) + 'miniconda3' + chr(92) + 'envs'),
    ('hostname', 'LAPTOP-ABC123XYZ'),
    ('posix home', '/home/someuser/project'),
    ('wsl mount', '/mnt/c/Users/someuser/x'),
]


@pytest.mark.parametrize('label,sample', BAD_MACHINE_SAMPLES)
def test_machine_leak_detector_fires(label, sample):
    text = _strip_allowed(sample)
    fired = [name for name, pattern in MACHINE_LEAK_PATTERNS if pattern.search(text)]
    assert fired, f'the detector missed a {label} leak: {sample!r}'


BAD_SECRET_SAMPLES = [
    ('private key', '-----BEGIN RSA PRIVATE KEY-----'),
    ('aws key', 'AKIA' + 'A' * 16),
    ('github token', 'ghp_' + 'a' * 36),
    ('slack token', 'xoxb-' + '1' * 12),
    ('jwt', 'eyJ' + 'a' * 12 + '.' + 'b' * 12 + '.' + 'c' * 12),
    ('dsn', 'postgresql://admin:hunter2@db.internal:5432/app'),
]


@pytest.mark.parametrize('label,sample', BAD_SECRET_SAMPLES)
def test_secret_detector_fires(label, sample):
    text = _strip_allowed(sample)
    fired = [name for name, pattern in SECRET_PATTERNS if pattern.search(text)]
    assert fired, f'the detector missed a {label}: {sample!r}'


def test_allowlist_actually_suppresses_the_known_safe_values():
    """The allowlist must work, or the gate fires on the project itself."""
    safe = [
        'Kevin030226',
        'postgresql://fiximg:fiximg@localhost:5432/fiximg_test',
        'password: "pbkdf2_sha256$260000$REPLACE_WITH_SALT$REPLACE_WITH_HASH"',
        'source="$REPO/weights"  # <repo>/weights',
    ]
    offenders = []
    for sample in safe:
        text = _strip_allowed(sample)
        for name, pattern in SECRET_PATTERNS + MACHINE_LEAK_PATTERNS:
            if pattern.search(text):
                offenders.append(f'{name} fired on allowlisted {sample[:30]!r}')
    assert not offenders, offenders


def test_pollution_detector_fires_on_an_unanchored_rule():
    """Prove the anchoring assertion is not vacuous.

    Feed `_matches` the exact unanchored shape that caused the historical
    `storage/` defect and require the source file to be reported as ignored.
    """
    unanchored = ['undefined', '~']
    assert _matches('src/fiximg/undefined/module.py', unanchored), (
        'an unanchored rule should match at depth -- if it does not, the'
        ' anchoring test above is not testing what it claims'
    )
    anchored = ['/undefined', '/~']
    assert not _matches('src/fiximg/undefined/module.py', anchored), (
        'a root-anchored rule must not match at depth'
    )


def test_evidence_ignore_detector_fires_on_xml_and_log():
    """The evidence rule must catch the extensions it names, and only those."""
    git = _entries(GITIGNORE)
    assert _matches('reports/evidence/junit.xml', git)
    assert _matches('reports/evidence/run.log', git)
    assert not _matches('reports/evidence/trivy_worker_result.txt', git)
    assert not _matches('reports/p5_publication_safety_remediation.md', git), (
        'top-level reports must stay committable after redaction'
    )
