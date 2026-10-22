"""Server-rendered page shells for the native web UI (plan §3.9).

The page is a document, not a component tree: the server renders the structure and
the browser fills it in. Two consequences shape this module:

* **Every dynamic value is escaped here.** `html.escape` is applied at the point
  the value enters the document, not left to the client, because the client is
  JavaScript that assigns `textContent` — and a template that ships unescaped text
  into the HTML is one `innerHTML` away from injecting it. The user-controlled
  strings on this page are the username, task ids, stage names and error messages.
* **The page works without JavaScript for reading.** The shell renders the sign-in
  form server-side and the panels are filled from the JSON routes, so a browser with
  scripting off can still sign in and see why; the interactive parts (upload,
  polling, before/after) are progressive enhancement.

Styling is inline in one style block rather than a CSS file. Not an oversight: the
app is served by a single process from a wheel that ships `src/fiximg` only, so a
separate stylesheet would be a file the wheel has to remember to package — the
exact class of packaging bug that makes a published image render unstyled. One
document with no external references also means no CDN, no font fetch, and nothing
to fail behind a firewall.
"""
from __future__ import annotations

import html
import json

#: Page title, kept identical to the auth pages so a user sees one product name.
from fiximg.ui.auth_pages import AUTH_PAGE_TITLE

#: Inline stylesheet. Dark, matching `auth_pages.build_auth_page_css`, so the
#: sign-in page and the app do not look like two different products.
PAGE_CSS = """
:root{color-scheme:dark}
*{box-sizing:border-box}
html,body{min-height:100%;margin:0}
body{background:#0f0f11;color:#f4f4f5;
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
 font-size:15px;line-height:1.5}
a{color:#fb923c}
a:hover{color:#fdba74}
.app{max-width:1200px;margin:0 auto;padding:18px 16px 64px}
.topbar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;
 justify-content:space-between;padding-bottom:14px;border-bottom:1px solid #27272a;
 margin-bottom:18px}
.brand{font-size:17px;font-weight:700;margin:0;line-height:1.35}
.who{font-size:13px;color:#a1a1aa;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:18px}
.tabs button{background:#18181b;border:1px solid #3f3f46;color:#e4e4e7;
 padding:8px 14px;border-radius:999px;font-size:14px;cursor:pointer}
.tabs button[aria-selected="true"]{background:#ea580c;border-color:#ea580c;color:#fff;
 font-weight:700}
.tabs button:focus-visible,.btn:focus-visible,input:focus-visible,select:focus-visible,
textarea:focus-visible,a:focus-visible{outline:2px solid #fb923c;outline-offset:2px}
.panel[hidden]{display:none}
.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px}
@media (max-width:860px){.grid{grid-template-columns:minmax(0,1fr)}}
.card{background:#18181b;border:1px solid #27272a;border-radius:12px;padding:16px}
.card h2,.card h3{margin:0 0 12px;font-size:15px;color:#e4e4e7}
.muted{color:#a1a1aa;font-size:13px}
.error{background:#7f1d1d;color:#fecaca;border-radius:8px;padding:10px 12px;
 margin:10px 0;white-space:pre-wrap}
.ok{background:#14532d;color:#86efac;border-radius:8px;padding:10px 12px;margin:10px 0}
.warn{background:#431407;color:#fdba74;border:1px solid #f97316;border-radius:8px;
 padding:10px 12px;margin:10px 0;font-weight:600}
.field{display:block;margin:12px 0}
.field span{display:block;font-size:13px;font-weight:600;color:#e4e4e7;margin-bottom:6px}
input[type=text],input[type=password],input[type=file],select,textarea{
 width:100%;padding:9px 11px;border:1px solid #3f3f46;border-radius:8px;
 background:#27272a;color:#f4f4f5;font-size:14px;font-family:inherit}
textarea{min-height:74px;resize:vertical}
.btn{display:inline-block;padding:9px 18px;border-radius:8px;border:1px solid #ea580c;
 background:#ea580c;color:#fff;font-size:14px;font-weight:700;cursor:pointer}
.btn:hover{background:#c2410c;border-color:#c2410c}
.btn[disabled]{opacity:.55;cursor:not-allowed}
.btn.secondary{background:#27272a;border-color:#3f3f46;color:#e4e4e7}
.btn.danger{background:#b91c1c;border-color:#b91c1c}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.switches{display:flex;flex-direction:column;gap:8px;margin:12px 0}
.switches label{display:flex;gap:8px;align-items:flex-start;font-size:14px}
.switches input{margin-top:3px}
.progress{margin:12px 0}
.bar{height:10px;background:#27272a;border-radius:999px;overflow:hidden}
.bar>i{display:block;height:100%;background:#ea580c;width:0;transition:width .2s}
.status{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;
 white-space:pre-wrap;word-break:break-word;background:#09090b;border:1px solid #27272a;
 border-radius:8px;padding:10px;margin:12px 0;min-height:3em}
.stages{list-style:none;padding:0;margin:8px 0;font-size:13px}
.stages li{padding:2px 0;color:#d4d4d8}
.stages li .why{color:#a1a1aa}
.preview{max-width:100%;border-radius:10px;border:1px solid #27272a;display:block}
.compare{position:relative;overflow:hidden;border-radius:10px;border:1px solid #27272a;
 max-width:100%;touch-action:none}
.compare img{display:block;width:100%;height:auto;user-select:none}
.compare .after{position:absolute;inset:0;overflow:hidden}
.compare .after img{position:absolute;inset:0;height:100%;object-fit:cover}
.compare .handle{position:absolute;top:0;bottom:0;width:2px;background:#fb923c;
 cursor:ew-resize}
.compare .handle::after{content:"";position:absolute;top:50%;left:50%;width:22px;
 height:22px;margin:-11px 0 0 -11px;border-radius:50%;background:#fb923c;
 border:2px solid #18181b}
.range{width:100%}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #27272a;
 vertical-align:top}
th{color:#a1a1aa;font-weight:600;font-size:12px;text-transform:uppercase;
 letter-spacing:.04em}
tr[aria-selected="true"]{background:#27272a}
td.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px}
.gallery button{padding:0;border:1px solid #27272a;border-radius:8px;overflow:hidden;
 background:#09090b;cursor:pointer;aspect-ratio:1}
.gallery img{width:100%;height:100%;object-fit:cover;display:block}
.gallery button[aria-current="true"]{border-color:#fb923c}
.examples{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}
.examples button{padding:0;width:44px;height:44px;border-radius:6px;overflow:hidden;
 border:1px solid #3f3f46;background:#09090b;cursor:pointer}
.examples img{width:100%;height:100%;object-fit:cover;display:block}
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;
 clip:rect(0,0,0,0);white-space:nowrap;border:0}
footer{margin-top:28px;color:#52525b;font-size:12px}
"""
#: The sign-in page. Rendered for an unauthenticated request to `/`; the Gradio
#: shell produced its own login form, so this keeps the same flow (form POST to
#: `/ui/login`) with the same fields.
SIGN_IN_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign in - {title}</title>
<style>{css}
.auth-page{{width:min(440px,calc(100vw - 32px));margin:0 auto;padding:32px 0}}
.auth-brand{{margin:0 0 20px;text-align:center;font-size:20px;line-height:1.35;
 font-weight:700;color:#fff}}
.auth-panel{{width:100%;padding:28px;background:#27272a;border:1px solid #3f3f46;
 border-radius:12px;box-shadow:0 16px 40px rgba(0,0,0,.45)}}
.auth-title{{margin:0 0 20px;text-align:center;font-size:18px;font-weight:700;color:#fff}}
.auth-label{{display:block;margin:14px 0 6px;font-size:14px;font-weight:600;color:#e4e4e7}}
.auth-input{{width:100%;height:42px;padding:0 12px;border:1px solid #3f3f46;
 border-radius:8px;background:#27272a;font-size:15px;color:#f4f4f5}}
.auth-input:focus{{outline:none;border-color:#fb923c;
 box-shadow:0 0 0 3px rgba(251,146,60,.25)}}
.auth-button{{width:100%;height:42px;margin-top:20px;border:1px solid #ea580c;
 border-radius:8px;background:#ea580c;color:#fff;font-size:15px;font-weight:700;
 cursor:pointer}}
.auth-button:hover{{background:#c2410c;border-color:#c2410c}}
.auth-status{{margin-bottom:14px;padding:10px 12px;border-radius:8px;font-size:14px}}
.auth-status.error{{background:#7f1d1d;color:#fecaca}}
.auth-status.success{{background:#14532d;color:#86efac}}
.auth-tips{{margin:12px 0 0;color:#71717a;font-size:13px}}
.auth-link{{display:block;margin-top:16px;text-align:center;font-size:14px;
 color:#fb923c;text-decoration:none;font-weight:600}}
</style>
</head>
<body>
<main class="auth-page">
<h1 class="auth-brand">{title}</h1>
<section class="auth-panel">
<h2 class="auth-title">Sign in</h2>
{status}
<form method="post" action="/ui/login">
<label class="auth-label" for="username">Username</label>
<input class="auth-input" id="username" name="username" value="{username}"
 autocomplete="username" required autofocus>
<label class="auth-label" for="password">Password</label>
<input class="auth-input" id="password" name="password" type="password"
 autocomplete="current-password" required>
<button class="auth-button" type="submit">Sign in</button>
</form>
<p class="auth-tips">No account yet?
<a class="auth-link" href="/register">Sign up</a></p>
</section>
</main>
</body>
</html>
"""

#: The application shell's static frame. `{panels}` is built from the pipeline mode
#: table rather than written out five times: the five restore/detect/colorize tabs
#: differ only in their captions, their example directory and whether they offer
#: option switches, so a literal copy of each would be five places to forget.
APP_FRAME = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<div class="app">
<header class="topbar">
<h1 class="brand">{title}</h1>
<div class="who">
<span>Signed in as <strong>{username}</strong></span>
<span class="muted">{role}</span>
<button class="btn secondary" type="button" id="sign-out">Sign out</button>
</div>
</header>

<div id="must-change" class="warn" hidden>
Your account still uses the initial one-time password. Set a new password under
<strong>Admin &rarr; Account</strong>; task submission stays locked until you do.
</div>
<div id="flash" role="status" aria-live="polite"></div>

<nav class="tabs" role="tablist" aria-label="Sections">
{tabs}
</nav>

<main>
{panels}
{history_panel}
{admin_panel}
</main>

<footer>Old Photo Restoration &amp; Scratch Repair - native web interface.</footer>
</div>
<script id="fiximg-bootstrap" type="application/json">{bootstrap}</script>
<script>{script}</script>
</body>
</html>
"""
#: One mode panel. Built once per entry in the pipeline mode table, so the tabs and
#: the handler's option names cannot disagree about which mode they belong to.
MODE_PANEL = """
<section class="panel" id="panel-{key}" role="tabpanel" aria-labelledby="tab-{key}"
 data-mode="{mode}"{hidden}>
<h2 class="sr-only">{tab_label}</h2>
<div class="grid">
<div class="card">
<h2>{input_label}</h2>
<div class="examples" data-examples="{examples}"></div>
<label class="field"><span>Choose an image (PNG or JPEG, up to {max_upload_mb} MB)</span>
<input type="file" accept="image/png,image/jpeg,image/webp" data-upload></label>
<div class="muted" data-file-name></div>{analysis}{switches}
<div class="row">
<button class="btn secondary" type="button" data-clear>{clear_label}</button>
<button class="btn" type="button" data-submit>{submit_label}</button>
</div>
</div>
<div class="card">
<h2>{result_label}</h2>
<div data-result>
<img class="preview" data-result-img alt="{result_label}" hidden>
</div>
<div class="compare" data-compare hidden>
<img data-compare-before alt="Original">
<div class="after"><img data-compare-after alt="{after_alt}"></div>
<div class="handle" tabindex="0" role="slider" aria-label="{slider_label}"
 aria-valuemin="0" aria-valuemax="100" aria-valuenow="50"></div>
</div>
<div class="progress"><div class="bar"><i data-bar></i></div></div>
<textarea class="status" data-status readonly aria-label="{status_label}"></textarea>
<ul class="stages" data-stages></ul>
<div class="row">
<a class="btn secondary" data-download hidden download>Download result</a>
</div>
</div>
</div>
</section>
"""

#: The analysis row, only on the mode whose plan is derived from the picture.
ANALYSIS_ROW = """
<textarea class="status" data-analysis readonly
 aria-label="Auto analysis (before you submit)"></textarea>
"""

#: The option checkboxes, built from the mode's declared switches.
SWITCH_BLOCK = """
<div class="switches" data-switches="{mode}">{boxes}</div>
"""

#: One checkbox. The caption comes from `pipeline_modes.switch_label`, so the
#: analysis-derived tab's wording ("only when the analysis finds a face") cannot be
#: silently reused on the restoration tabs, where it would promise a stage the box
#: does not schedule.
SWITCH_BOX = """
<label><input type="checkbox" data-switch="{name}"{checked}>
<span>{caption}</span></label>
"""
#: The history panel: a page of rows, thumbnails, the before/after slider, the four
#: preview slots (§3.9.2) and the two re-run controls.
HISTORY_PANEL = """
<section class="panel" id="panel-history" role="tabpanel" aria-labelledby="tab-history"
 hidden>
<h2 class="sr-only">History</h2>
<div class="card">
<div class="row">
<button class="btn secondary" type="button" id="history-refresh">Refresh</button>
<label class="row" style="gap:6px"><span class="muted">Rows per page</span>
<select id="history-page-size" style="width:auto">
<option>10</option><option selected>20</option><option>50</option>
<option>100</option></select></label>
<button class="btn secondary" type="button" id="history-prev">&lt; Prev</button>
<span class="muted" id="history-page-label">Page 1 of 1</span>
<button class="btn secondary" type="button" id="history-next">Next &gt;</button>
<button class="btn" type="button" id="history-rerun" disabled>Re-run selected</button>
<button class="btn secondary" type="button" id="history-retry" disabled>
Retry failed (same task)</button>
</div>
<p class="muted" id="history-status" role="status" aria-live="polite"></p>
<div class="gallery" id="history-gallery"></div>
<table>
<caption class="sr-only">Recent tasks; select a row to preview it</caption>
<thead><tr>
<th scope="col">Task</th><th scope="col">Type</th><th scope="col">Created</th>
<th scope="col">Duration</th><th scope="col">Status</th><th scope="col">Progress</th>
<th scope="col">Metrics</th>
</tr></thead>
<tbody id="history-rows"></tbody>
</table>
<h3>Selected task</h3>
<div class="compare" id="history-compare" hidden>
<img data-compare-before alt="Original">
<div class="after"><img data-compare-after alt="Restored"></div>
<div class="handle" tabindex="0" role="slider"
 aria-label="Original and restored slider"
 aria-valuemin="0" aria-valuemax="100" aria-valuenow="50"></div>
</div>
<div class="row" style="margin-top:10px">
<figure style="margin:0;flex:1 1 120px">
<img class="preview" id="history-input" alt="Original" hidden>
<figcaption class="muted">Original</figcaption>
</figure>
<figure style="margin:0;flex:1 1 120px">
<img class="preview" id="history-result" alt="Restored" hidden>
<figcaption class="muted">Restored</figcaption>
</figure>
<figure style="margin:0;flex:1 1 120px">
<img class="preview" id="history-mask" alt="Mask" hidden>
<figcaption class="muted">Mask</figcaption>
</figure>
<figure style="margin:0;flex:1 1 120px">
<img class="preview" id="history-face" alt="Face crop" hidden>
<figcaption class="muted">Face crop</figcaption>
</figure>
</div>
<textarea class="status" id="history-detail" readonly aria-label="Task detail"></textarea>
<a class="btn secondary" id="history-download" hidden download>Download this result</a>
</div>
</section>
"""
#: The admin panel: the four Gradio tabs (Account, Task Management, Photo Archive,
#: User Management) plus Models, which the API has always exposed and the Gradio
#: panel never reached.
ADMIN_PANEL = """
<section class="panel" id="panel-admin" role="tabpanel" aria-labelledby="tab-admin"
 hidden>
<h2 class="sr-only">Admin</h2>
<div class="tabs" role="tablist" aria-label="Admin sections">
<button role="tab" type="button" data-admin-tab="account" aria-selected="true"
 aria-controls="admin-account" id="atab-account">Account</button>
<button role="tab" type="button" data-admin-tab="tasks" aria-selected="false"
 aria-controls="admin-tasks" id="atab-tasks">Task management</button>
<button role="tab" type="button" data-admin-tab="archive" aria-selected="false"
 aria-controls="admin-archive" id="atab-archive">Photo archive</button>
<button role="tab" type="button" data-admin-tab="users" aria-selected="false"
 aria-controls="admin-users" id="atab-users">User management</button>
<button role="tab" type="button" data-admin-tab="models" aria-selected="false"
 aria-controls="admin-models" id="atab-models">Models</button>
</div>

<section id="admin-account" role="tabpanel" aria-labelledby="atab-account">
<div class="card">
<h2>Account / change password</h2>
<form id="password-form">
<label class="field"><span>New password</span>
<input type="password" name="new_password" autocomplete="new-password" required></label>
<label class="field"><span>Confirm new password</span>
<input type="password" name="confirm_password" autocomplete="new-password"
 required></label>
<button class="btn" type="submit">Change my password</button>
</form>
<p class="muted" id="password-status" role="status" aria-live="polite"></p>
</div>
</section>

<section id="admin-tasks" role="tabpanel" aria-labelledby="atab-tasks" hidden>
<div class="card">
<h2>Task processing history</h2>
<div class="row"><button class="btn secondary" type="button" id="admin-history-refresh">
Refresh</button></div>
<table>
<caption class="sr-only">Every task, newest first</caption>
<thead><tr><th scope="col">Time</th><th scope="col">User</th>
<th scope="col">Task type</th><th scope="col">Status</th><th scope="col">PSNR</th>
<th scope="col">SSIM</th><th scope="col">MAE</th><th scope="col">Input file</th></tr></thead>
<tbody id="admin-history-rows"></tbody>
</table>
<h3>Statistics overview</h3>
<textarea class="status" id="admin-stats" readonly aria-label="Statistics"></textarea>
<h3>Danger zone</h3>
<p class="muted">Clears the legacy V1 history rows. Task records are the queue's
state and are kept.</p>
<button class="btn danger" type="button" id="admin-clear-history">
Clear legacy records</button>
<p class="muted" id="admin-clear-status" role="status" aria-live="polite"></p>
</div>
</section>

<section id="admin-archive" role="tabpanel" aria-labelledby="atab-archive" hidden>
<div class="card">
<h2>Photo archive</h2>
<div class="row"><button class="btn secondary" type="button" id="archive-refresh">
Refresh</button></div>
<label class="field"><span>Select a record</span>
<select id="archive-select"></select></label>
<div class="row">
<figure style="margin:0;flex:1 1 160px">
<img class="preview" id="archive-input" alt="Original image" hidden>
<figcaption class="muted">Original</figcaption>
</figure>
<figure style="margin:0;flex:1 1 160px">
<img class="preview" id="archive-output" alt="Restored image" hidden>
<figcaption class="muted">Restored</figcaption>
</figure>
</div>
<textarea class="status" id="archive-detail" readonly
 aria-label="Processing info"></textarea>
</div>
</section>

<section id="admin-users" role="tabpanel" aria-labelledby="atab-users" hidden>
<div class="card">
<h2>Current users</h2>
<div class="row"><button class="btn secondary" type="button" id="users-refresh">
Refresh user list</button></div>
<table>
<caption class="sr-only">Current users</caption>
<thead><tr><th scope="col">Username</th><th scope="col">Role</th></tr></thead>
<tbody id="users-rows"></tbody>
</table>
<h3>Add user</h3>
<form id="user-add-form">
<label class="field"><span>New username</span>
<input type="text" name="username" autocomplete="off" required></label>
<label class="field"><span>Password</span>
<input type="password" name="password" autocomplete="new-password" required></label>
<label class="field"><span>Role</span>
<select name="role"><option value="user" selected>user</option>
<option value="admin">admin</option></select></label>
<button class="btn" type="submit">Add user</button>
</form>
<p class="muted" id="user-add-status" role="status" aria-live="polite"></p>
<h3>Change password / role</h3>
<form id="user-edit-form">
<label class="field"><span>Select user</span>
<select name="username" id="user-edit-select"></select></label>
<label class="field"><span>New password (leave empty to keep)</span>
<input type="password" name="password" autocomplete="new-password"></label>
<label class="field"><span>New role (leave empty to keep)</span>
<select name="role"><option value="" selected>(keep)</option>
<option value="user">user</option><option value="admin">admin</option></select></label>
<button class="btn" type="submit">Save changes</button>
</form>
<p class="muted" id="user-edit-status" role="status" aria-live="polite"></p>
<h3>Delete user</h3>
<form id="user-delete-form">
<label class="field"><span>Select user</span>
<select name="username" id="user-delete-select"></select></label>
<button class="btn danger" type="submit">Delete user</button>
</form>
<p class="muted" id="user-delete-status" role="status" aria-live="polite"></p>
</div>
</section>

<section id="admin-models" role="tabpanel" aria-labelledby="atab-models" hidden>
<div class="card">
<h2>Models</h2>
<div class="row"><button class="btn secondary" type="button" id="models-refresh">
Refresh</button></div>
<table>
<caption class="sr-only">Models this deployment declares and can run</caption>
<thead><tr><th scope="col">Name</th><th scope="col">Version</th>
<th scope="col">Status</th><th scope="col">Device</th>
<th scope="col">Weights</th><th scope="col">Actions</th></tr></thead>
<tbody id="models-rows"></tbody>
</table>
<p class="muted" id="models-status" role="status" aria-live="polite"></p>
</div>
</section>
</section>
"""
#: 404 for a path the UI does not serve. Deliberately minimal: an unmatched path
#: must not be answered with the application shell, or a typo would look like a
#: working page.
NOT_FOUND_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Not found</title>
<style>{css}</style>
</head>
<body>
<div class="app">
<h1 class="brand">Not found</h1>
<p class="muted">There is nothing at <code>{path}</code>.</p>
<p><a href="/">Back to the application</a></p>
</div>
</body>
</html>
"""

#: How each tab presents itself. Keyed by the pipeline mode it submits, because the
#: tab and the handler that reads its switches have to name the same mode.
MODE_TABS = {
    "auto": {
        "tab_label": "Auto Restore",
        "tab_caption": "Auto Restore",
        "input_label": "Image to restore (any old photo)",
        "result_label": "Restored result",
        "after_alt": "Restored",
        "slider_label": "Before and after slider: original and restored",
        "status_label": "Evaluation and pipeline decisions",
        "submit_label": "Start auto restore",
        "clear_label": "Clear all",
        "examples": "all",
    },
    "restore": {
        "tab_label": "Restore old photo (no scratches)",
        "tab_caption": "Restore (no scratches)",
        "input_label": "Image to restore",
        "result_label": "Restored result",
        "after_alt": "Restored",
        "slider_label": "Before and after slider: original and restored",
        "status_label": "Difference metrics vs. degraded original (PSNR/SSIM/MAE)",
        "submit_label": "Start restoration",
        "clear_label": "Clear image",
        "examples": "old",
    },
    "restore_scratch": {
        "tab_label": "Restore old photo (with scratches)",
        "tab_caption": "Restore (scratches)",
        "input_label": "Image to restore",
        "result_label": "Restored result",
        "after_alt": "Restored",
        "slider_label": "Before and after slider: original and restored",
        "status_label": "Difference metrics vs. degraded original (PSNR/SSIM/MAE)",
        "submit_label": "Start scratch repair",
        "clear_label": "Clear all",
        "examples": "old_w_scratch",
    },
    "detect": {
        "tab_label": "Scratch detection",
        "tab_caption": "Scratch detection",
        "input_label": "Image to detect",
        "result_label": "Detection result (scratch mask)",
        "after_alt": "Detected mask",
        "slider_label": "Original and detected mask slider",
        "status_label": "Detection status",
        "submit_label": "Detect scratches",
        "clear_label": "Clear all",
        "examples": "old_w_scratch",
    },
    "colorize": {
        "tab_label": "Old photo colorization",
        "tab_caption": "Colorize",
        "input_label": "Image to colorize (black and white)",
        "result_label": "Colorized result",
        "after_alt": "Colorized",
        "slider_label": "Before and after slider: original and colorized",
        "status_label": "Colorization status",
        "submit_label": "Start colorization",
        "clear_label": "Clear all",
        "examples": "color",
    },
}

#: The order the tabs appear in. The mode table's own order, so a new mode lands in
#: the strip instead of being built and never shown.
TAB_ORDER = ("auto", "restore", "restore_scratch", "detect", "colorize")

#: Which tab is selected first. The Gradio shell landed an administrator on the
#: Admin tab and an ordinary user on Auto Restore, because the default first tab was
#: the one an administrator could not use; the same rule is applied here at render
#: time so the page is correct before any script runs.
FIRST_TAB_ADMIN = "admin"
FIRST_TAB_USER = "auto"


def _escape(value) -> str:
    """Escape one value for HTML text or a double-quoted attribute."""
    return html.escape("" if value is None else str(value), quote=True)


def _json_for_script(payload) -> str:
    """Embed a JSON payload inside a script element of type application/json.

    A literal closing script tag inside a JSON string would end the element early,
    so the two characters that can do it are escaped as their JSON unicode form.
    The result is still valid JSON, and JSON.parse turns it back into the original
    text — which is what lets the browser read the bootstrap without the server
    having to build JavaScript source out of user data.
    """
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return text.replace("</", "<" + chr(92) + "/")


def render_switch_block(mode: str) -> str:
    """The option checkboxes for one mode, or nothing when it offers none.

    The captions come from `pipeline_modes.switch_label`, which is the one place
    that knows the Auto tab words the same key differently. The initial state is the
    declared default, so a submit with nobody touching a box sends what the planner
    would have chosen anyway.
    """
    from fiximg.application.pipeline_modes import SWITCHES, switch_label

    declared = SWITCHES.get(mode) or {}
    if not declared:
        return ""
    boxes = "".join(
        SWITCH_BOX.format(
            name=_escape(name),
            caption=_escape(switch_label(mode, name)),
            checked=" checked" if default else "",
        )
        for name, default in declared.items()
    )
    return SWITCH_BLOCK.format(mode=_escape(mode), boxes=boxes)


def render_mode_panel(mode: str, *, selected: bool, max_upload_mb: int) -> str:
    """One mode tab: its panel, its examples and its switches."""
    from fiximg.application.pipeline_modes import ANALYSIS_DECIDED_MODES

    spec = MODE_TABS[mode]
    return MODE_PANEL.format(
        key=_escape(mode),
        mode=_escape(mode),
        hidden="" if selected else " hidden",
        tab_label=_escape(spec["tab_label"]),
        input_label=_escape(spec["input_label"]),
        result_label=_escape(spec["result_label"]),
        after_alt=_escape(spec["after_alt"]),
        slider_label=_escape(spec["slider_label"]),
        status_label=_escape(spec["status_label"]),
        submit_label=_escape(spec["submit_label"]),
        clear_label=_escape(spec["clear_label"]),
        examples=_escape(spec["examples"]),
        max_upload_mb=_escape(max_upload_mb),
        analysis=ANALYSIS_ROW if mode in ANALYSIS_DECIDED_MODES else "",
        switches=render_switch_block(mode),
    )


def render_tab_button(mode: str, *, selected: bool) -> str:
    """One button in the top tab strip."""
    spec = MODE_TABS[mode]
    return (
        f'<button role="tab" type="button" data-tab="{_escape(mode)}" '
        f'aria-selected="{"true" if selected else "false"}" '
        f'aria-controls="panel-{_escape(mode)}" id="tab-{_escape(mode)}">'
        f'{_escape(spec["tab_caption"])}</button>'
    )


#: The two tabs that are not pipeline modes: History for everyone, Admin only for an
#: administrator.
FIXED_TABS = {"history": "History", "admin": "Admin"}


def render_fixed_tab(key: str, caption: str, *, selected: bool) -> str:
    """One of the two non-mode tabs.

    Kept separate from :func:`render_tab_button` because those two are built from the
    pipeline mode table and these are not a mode: they submit nothing.
    """
    return (
        f'<button role="tab" type="button" data-tab="{_escape(key)}" '
        f'aria-selected="{"true" if selected else "false"}" '
        f'aria-controls="panel-{_escape(key)}" id="tab-{_escape(key)}">'
        f"{_escape(caption)}</button>"
    )


def render_sign_in(*, error: str = "", username: str = "") -> str:
    """The sign-in document; `error` is escaped into the status block."""
    status = f'<div class="auth-status error">{_escape(error)}</div>' if error else ""
    return SIGN_IN_HTML.format(
        title=_escape(AUTH_PAGE_TITLE),
        css=PAGE_CSS,
        status=status,
        username=_escape(username),
    )


def render_app(*, username: str, role: str, bootstrap: dict, script: str) -> str:
    """The application document for a signed-in caller.

    `username` and `role` are escaped; `bootstrap` is embedded as JSON (never as
    JavaScript source) and `script` is the static UI script, which is the only code
    in the page and contains no interpolated values.
    """
    from fiximg.config import settings

    is_admin = role == "admin"
    first = FIRST_TAB_ADMIN if is_admin else FIRST_TAB_USER
    tabs = "".join(
        render_tab_button(mode, selected=(mode == first)) for mode in TAB_ORDER
    )
    # The History tab is rendered for every caller; the Admin tab only for an
    # administrator. Rendering the button server-side is what makes the layout a
    # fact rather than a client-side courtesy: a user's document simply does not
    # contain the Admin button, so there is nothing to unhide.
    tabs += render_fixed_tab("history", "History", selected=(first == "history"))
    if is_admin:
        tabs += render_fixed_tab("admin", "Admin", selected=(first == "admin"))
    panels = "".join(
        render_mode_panel(mode, selected=(mode == first),
                          max_upload_mb=settings.max_upload_mb)
        for mode in TAB_ORDER
    )
    return APP_FRAME.format(
        title=_escape(AUTH_PAGE_TITLE),
        css=PAGE_CSS,
        username=_escape(username),
        role=_escape(role),
        tabs=tabs,
        panels=panels,
        history_panel=HISTORY_PANEL,
        # The admin panel's markup is only in an administrator's document. A hidden
        # panel would still ship every admin form (and every route name) to a user
        # who is not allowed to use it; an empty string ships nothing.
        admin_panel=ADMIN_PANEL if is_admin else "",
        bootstrap=_json_for_script(bootstrap),
        script=script,
    )


def render_not_found(path: str) -> str:
    """The 404 document; `path` is escaped because it is caller-controlled."""
    return NOT_FOUND_HTML.format(css=PAGE_CSS, path=_escape(path))


__all__ = [
    "ADMIN_PANEL",
    "APP_FRAME",
    "FIRST_TAB_ADMIN",
    "FIRST_TAB_USER",
    "HISTORY_PANEL",
    "MODE_TABS",
    "NOT_FOUND_HTML",
    "PAGE_CSS",
    "SIGN_IN_HTML",
    "TAB_ORDER",
    "FIXED_TABS",
    "render_app",
    "render_fixed_tab",
    "render_mode_panel",
    "render_not_found",
    "render_sign_in",
    "render_switch_block",
    "render_tab_button",
]
