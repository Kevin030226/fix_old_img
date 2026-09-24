"""Login/registration page rendering (extracted from V1 main.py, unchanged UX)."""
import html

APP_TITLE = (
    "Old Photo Restoration & Scratch Repair via Deep Learning "
    "(GANs and Variational Autoencoders)"
)
AUTH_PAGE_TITLE = APP_TITLE


def build_auth_page_css(center_body=True):
    body_layout = (
        "display:flex!important;flex-direction:column!important;"
        "justify-content:center!important;align-items:center!important;"
        if center_body
        else ""
    )
    return (
        "<style>"
        "html{color-scheme:dark}"
        "*{box-sizing:border-box}"
        "html,body{min-height:100%;margin:0}"
        "body{"
        f"{body_layout}"
        "background:#0f0f11!important;"
        "font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif;"
        "color:#f4f4f5;"
        "}"
        ".auth-page{width:min(440px,calc(100vw - 32px));margin:0 auto;padding:32px 0}"
        ".auth-brand{margin:0 0 20px;text-align:center;font-size:22px;line-height:1.35;font-weight:700;color:#ffffff}"
        ".auth-panel{width:100%;padding:28px;background:#27272a;border:1px solid #3f3f46;border-radius:12px;"
        "box-shadow:0 1px 2px rgba(0,0,0,.3),0 16px 40px rgba(0,0,0,.45)}"
        ".auth-title{margin:0 0 20px;text-align:center;font-size:20px;line-height:1.4;font-weight:700;color:#ffffff}"
        ".auth-label{display:block;margin:14px 0 6px;font-size:14px;font-weight:600;color:#e4e4e7}"
        ".auth-input{width:100%;height:42px;padding:0 12px;border:1px solid #3f3f46;border-radius:8px;"
        "background:#27272a;font-size:15px;color:#f4f4f5;transition:border-color .15s,box-shadow .15s}"
        ".auth-input::placeholder{color:#71717a}"
        ".auth-input:focus{outline:none;border-color:#fb923c;box-shadow:0 0 0 3px rgba(251,146,60,.25)}"
        ".auth-button{width:100%;height:42px;margin-top:20px;border:1px solid #ea580c;border-radius:8px;"
        "background:#ea580c;color:#ffffff;font-size:15px;"
        "font-weight:700;cursor:pointer;transition:filter .15s,border-color .15s}"
        ".auth-button:hover{background:#c2410c;border-color:#c2410c}"
        ".auth-status{margin-bottom:14px;padding:10px 12px;border-radius:8px;font-size:14px;line-height:1.5}"
        ".auth-status.success{background:#14532d;color:#86efac}"
        ".auth-status.error{background:#7f1d1d;color:#fecaca}"
        ".auth-tips{margin:12px 0 0;color:#71717a;font-size:13px;line-height:1.6}"
        ".auth-link{display:block;margin-top:16px;text-align:center;font-size:14px;color:#fb923c;"
        "text-decoration:none;font-weight:600}"
        ".auth-link:hover{color:#fdba74}"
        ".register-entry{margin-top:16px;font-size:14px;color:#71717a;text-align:center}"
        ".register-entry a{display:inline-block;margin-left:4px;padding:8px 22px;border:1px solid #ea580c;"
        "border-radius:999px;background:#ea580c;color:#ffffff;"
        "text-decoration:none;font-weight:700;box-shadow:0 1px 2px rgba(0,0,0,.3);"
        "transition:filter .15s,border-color .15s}"
        ".register-entry a:hover{background:#c2410c;border-color:#c2410c}"
        ".register-entry-floating{position:fixed;left:50%;bottom:28px;transform:translateX(-50%);"
        "z-index:9999;white-space:nowrap}"
        "@media (max-width:480px){.auth-page{width:calc(100vw - 24px);padding:16px 0}"
        ".auth-panel{padding:22px}.auth-brand{font-size:18px}}"
        "</style>"
    )


def render_register_page(message="", success=False, username=""):
    safe_message = html.escape(message)
    safe_username = html.escape(username)
    status_class = "success" if success else "error"
    status_html = (
        f'<div class="auth-status {status_class}">{safe_message}</div>' if message else ""
    )
    login_text = "Sign In" if success else "Back to Login"
    return f"""<!doctype html>
    <html lang="zh-CN">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>User Registration</title>
        {build_auth_page_css()}
    </head>
    <body>
        <main class="auth-page">
            <h1 class="auth-brand">{AUTH_PAGE_TITLE}</h1>
            <section class="auth-panel">
            <h2 class="auth-title">User Registration</h2>
            {status_html}
            <form method="post" action="/register">
                <label class="auth-label" for="username">Username</label>
                <input class="auth-input" id="username" name="username" value="{safe_username}"
                       autocomplete="username" required>
                <label class="auth-label" for="password">Password</label>
                <input class="auth-input" id="password" name="password" type="password"
                       autocomplete="new-password" required>
                <label class="auth-label" for="confirm_password">Confirm Password</label>
                <input class="auth-input" id="confirm_password" name="confirm_password" type="password"
                       autocomplete="new-password" required>
                <button class="auth-button" type="submit">Sign Up</button>
            </form>
            <p class="auth-tips">Username: 3-32 letters, digits, underscores or hyphens; password: at least 6 characters.</p>
            <a class="auth-link" href="/">{login_text}</a>
            </section>
        </main>
    </body>
    </html>
    """
