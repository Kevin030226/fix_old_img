"""HTTP middleware extracted from V1 main.py: login-page styling + role layout.

Role layout uses two complementary mechanisms:

* tab *content* is controlled server-side from the page-load handler
  (app/ui/gradio_app.apply_role) by updating each TabItem's `visible` prop;
* tab *buttons* are hidden here by a role stylesheet. Gradio's Tabs component
  does not rebuild its button list when `visible` changes at runtime, so a
  server-side update alone leaves the buttons on screen. Gradio attaches
  "{elem_id}-button" to the button element, which is what these rules target.

CSS is used rather than the previous JS DOM patching because Gradio re-renders
from server props and would wipe inline styles; a stylesheet keeps applying.
"""
import re

from fastapi import Request, Response

from app.repositories.user_repository import get_user
from app.ui.auth_pages import build_auth_page_css

_FORCE_DARK_SCRIPT = (
    "<script>"
    "(function(){try{"
    "var _mm=window.matchMedia.bind(window);"
    "window.matchMedia=function(q){"
    "var m=_mm(q);"
    "if(q&&String(q).indexOf('prefers-color-scheme')!==-1){"
    "try{Object.defineProperty(m,'matches',{configurable:true,get:function(){return true;}});"
    "m.addEventListener=function(){};m.removeEventListener=function(){};}catch(e){}}"
    "return m;};}catch(e){}})();"
    "</script>"
)

#: Admin sees only the Admin Panel: hide every function-tab button.
_ADMIN_TAB_CSS = (
    "<style>"
    "#tab_auto-button,#tab_restore-button,#tab_scratch-button,"
    "#tab_detect-button,#tab_colorize-button{display:none!important}"
    "</style>"
)

#: Users never see the Admin Panel.
_USER_TAB_CSS = "<style>#admin_panel-button{display:none!important}</style>"


def _gradio_request_user(request: Request, gradio_app):
    """Read the Gradio session; return the logged-in username, or None."""
    if gradio_app is None:
        return None
    cid = getattr(gradio_app, "cookie_id", None)
    if not cid:
        return None
    token = request.cookies.get(f"access-token-{cid}") or request.cookies.get(
        f"access-token-unsecure-{cid}"
    )
    if token and token in getattr(gradio_app, "tokens", {}):
        return gradio_app.tokens[token]
    return None


def _rebuild_html_response_headers(raw_headers):
    """Keep all original headers (incl. multiple Set-Cookie), minus content-length."""
    return [
        (k.decode("latin-1"), v.decode("latin-1"))
        for k, v in raw_headers
        if k.lower() != b"content-length"
    ]


def make_html_middleware(gradio_app_holder):
    """Build the inject_login_css middleware; gradio_app_holder is a dict with 'app'."""

    async def inject_login_css(request: Request, call_next):
        response = await call_next(request)
        if "text/html" not in response.headers.get("content-type", ""):
            return response

        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        if not body:
            return response

        gradio_app = gradio_app_holder.get("app")
        has_login_form = (
            b'name="username"' in body
            and b'name="password"' in body
            and b'type="password"' in body
        )
        is_login_page = (
            request.url.path == "/"
            and _gradio_request_user(request, gradio_app) is None
            and (has_login_form or not request.headers.get("authorization"))
        )

        if is_login_page:
            css_bytes = build_auth_page_css(center_body=has_login_form).encode("utf-8")
            entry_class = (
                "register-entry" if has_login_form else "register-entry register-entry-floating"
            )
            link_bytes = (
                f'<div class="{entry_class}">No account yet? <a href="/register">Sign up now</a></div>'
            ).encode()
            if b"</head>" in body:
                body = body.replace(b"</head>", css_bytes + b"</head>")
            elif b"<body" in body:
                body = re.sub(b"(<body[^>]*>)", css_bytes + b"\\1", body)
            if b"</body>" in body:
                body = body.replace(b"</body>", link_bytes + b"</body>")
            else:
                body += link_bytes

        # Role layout: hide the tab buttons this caller must not see. Runs for
        # the authenticated document only; the cookie is set by /login, so the
        # post-login page load already carries it.
        username = _gradio_request_user(request, gradio_app)
        if username and request.url.path == "/":
            user = get_user(username)
            if user:
                role_css = (
                    _ADMIN_TAB_CSS if user.get("role") == "admin" else _USER_TAB_CSS
                ).encode("utf-8")
                if b"</head>" in body:
                    body = body.replace(b"</head>", role_css + b"</head>")
                elif b"<body" in body:
                    body = re.sub(b"(<body[^>]*>)", role_css + b"\\1", body)

        dark_script_bytes = _FORCE_DARK_SCRIPT.encode("utf-8")
        if b"</head>" in body:
            body = body.replace(b"</head>", dark_script_bytes + b"</head>")
        elif b"<body" in body:
            body = re.sub(b"(<body[^>]*>)", dark_script_bytes + b"\\1", body)
        else:
            body = dark_script_bytes + body

        rebuilt = Response(content=body, status_code=response.status_code)
        rebuilt.raw_headers = [
            (k.encode("latin-1"), v.encode("latin-1"))
            for k, v in _rebuild_html_response_headers(response.raw_headers)
        ]
        return rebuilt

    return inject_login_css
