"""旧照片修复系统 —— Web 服务入口（Gradio 6 + FastAPI + SQLite 重写版）。

运行：
    python main.py
默认监听 http://127.0.0.1:9502（可用 FIXIMG_HOST / FIXIMG_PORT 覆盖）。
"""
import html
import os
import re
import sys
import time
from urllib.parse import parse_qs

import gradio as gr
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response

from app.db import add_user, get_user, init_db
from app.colorizer import run_colorize
from app.pipeline import PIPELINE_MODES, format_evaluation, run_pipeline
from config.ratelimit import (
    client_ip,
    register_global_limiter,
    register_ip_limiter,
    register_username_limiter,
)
from config.security import hash_password, verify_password
from config.weights_check import WeightsIntegrityError, verify_weights

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

APP_TITLE = "基于深度学习(GAN 与变分编码器 VAEs)的旧照片恢复与划痕修复算法模型"
AUTH_PAGE_TITLE = APP_TITLE


# ============================== 认证 ==============================
def auth_fn(username: str, password: str) -> bool:
    """登录校验：哈希比对（常量时间）。"""
    user = get_user(username)
    if not user:
        return False
    return verify_password(password, user.get("password", ""))


# ============================== 注册页面 ==============================
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
    login_text = "立即登录" if success else "返回登录"
    return f"""<!doctype html>
    <html lang="zh-CN">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>用户注册</title>
        {build_auth_page_css()}
    </head>
    <body>
        <main class="auth-page">
            <h1 class="auth-brand">{AUTH_PAGE_TITLE}</h1>
            <section class="auth-panel">
            <h2 class="auth-title">用户注册</h2>
            {status_html}
            <form method="post" action="/register">
                <label class="auth-label" for="username">用户名</label>
                <input class="auth-input" id="username" name="username" value="{safe_username}"
                       autocomplete="username" required>
                <label class="auth-label" for="password">密码</label>
                <input class="auth-input" id="password" name="password" type="password"
                       autocomplete="new-password" required>
                <label class="auth-label" for="confirm_password">确认密码</label>
                <input class="auth-input" id="confirm_password" name="confirm_password" type="password"
                       autocomplete="new-password" required>
                <button class="auth-button" type="submit">注册</button>
            </form>
            <p class="auth-tips">用户名支持 3-32 位字母、数字、下划线或短横线；密码至少 6 位。</p>
            <a class="auth-link" href="/">{login_text}</a>
            </section>
        </main>
    </body>
    </html>
    """


# ============================== FastAPI ==============================
app = FastAPI()


@app.get("/register", response_class=HTMLResponse)
async def register_page():
    return HTMLResponse(render_register_page())


@app.post("/register", response_class=HTMLResponse)
async def register_user(request: Request):
    ip = client_ip(request)
    if not register_ip_limiter.hit(ip):
        return HTMLResponse(
            render_register_page("注册过于频繁，请稍后再试（如仍有问题请联系管理员）"),
            status_code=429,
        )
    if not register_global_limiter.hit("__global__"):
        return HTMLResponse(
            render_register_page("当前注册人数过多，请稍后再试"), status_code=429
        )

    body = (await request.body()).decode("utf-8")
    form = parse_qs(body)
    username = form.get("username", [""])[0].strip()
    password = form.get("password", [""])[0]
    confirm_password = form.get("confirm_password", [""])[0]

    if username and not register_username_limiter.hit(username):
        return HTMLResponse(
            render_register_page("该用户名操作过于频繁，请稍后再试", username=username),
            status_code=429,
        )
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,32}", username):
        return HTMLResponse(render_register_page("用户名格式不正确", username=username))
    if len(password) < 6:
        return HTMLResponse(render_register_page("密码长度不能少于 6 位", username=username))
    if password != confirm_password:
        return HTMLResponse(render_register_page("两次输入的密码不一致", username=username))
    if get_user(username):
        return HTMLResponse(
            render_register_page(f"用户 '{username}' 已存在", username=username)
        )

    add_user(username, hash_password(password), "user")
    return HTMLResponse(
        render_register_page("注册成功，请返回登录页登录", success=True, username=username)
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


# ============================== 强制深色主题 ==============================
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


# 管理员界面精简：隐藏四个功能标签页并自动切到管理面板。
# Gradio 6 的 Tab.visible 动态更新不生效，故由服务端判定角色后注入脚本。
_ADMIN_UI_SCRIPT = (
    "<script>"
    "(function(){"
    "var labels=['不带划痕的旧照片复原模块','带划痕的旧照片复原模块',"
    "'划痕检测模块','老照片上色模块'];"
    "function adminMode(){"
    "var btns=Array.from(document.querySelectorAll('button'));"
    "var tabs=Array.from(document.querySelectorAll('[role=\"tab\"]'));"
    "btns.forEach(function(b){"
    "var t=(b.textContent||'').trim();"
    "if(labels.indexOf(t)>=0){b.style.display='none';}"
    "});"
    "tabs.forEach(function(tab){"
    "var t=(tab.textContent||'').trim();"
    "if(labels.indexOf(t)>=0){tab.style.display='none';}"
    "});"
    "var adminTab=tabs.find(function(tab){return (tab.textContent||'').trim()==='管理面板';});"
    "if(adminTab && adminTab.getAttribute('aria-selected')!=='true'){adminTab.click();}"
    "}"
    "if(document.readyState==='loading'){"
    "document.addEventListener('DOMContentLoaded',adminMode);"
    "}else{adminMode();}"
    "setTimeout(adminMode,400);setTimeout(adminMode,900);"
    "setTimeout(adminMode,1600);setTimeout(adminMode,3000);"
    "setTimeout(adminMode,5000);"
    "})();"
    "</script>"
)


# 普通用户界面精简：隐藏管理面板标签页。
_USER_UI_SCRIPT = (
    "<script>"
    "(function(){"
    "function hide(){"
    "var btns=Array.from(document.querySelectorAll('button'));"
    "var tabs=Array.from(document.querySelectorAll('[role=\"tab\"]'));"
    "btns.forEach(function(b){"
    "if((b.textContent||'').trim()==='管理面板'){b.style.display='none';}"
    "});"
    "tabs.forEach(function(tab){"
    "if((tab.textContent||'').trim()==='管理面板'){tab.style.display='none';}"
    "});"
    "}"
    "if(document.readyState==='loading'){"
    "document.addEventListener('DOMContentLoaded',hide);"
    "}else{hide();}"
    "setTimeout(hide,1000);setTimeout(hide,3000);"
    "})();"
    "</script>"
)


# 管理员界面 CSS：隐藏四个功能标签页并强制显示管理面板。
_ADMIN_UI_CSS = (
    "<style>"
    "#tab_restore,#tab_scratch,#tab_detect,#tab_colorize{display:none!important}"
    "#admin_panel{display:block!important}"
    "</style>"
)

# 普通用户界面 CSS：隐藏管理面板。
_USER_UI_CSS = (
    "<style>"
    "#admin_panel{display:none!important}"
    "</style>"
)


def _gradio_request_user(request):
    """读取 Gradio 会话，返回当前登录用户名；未登录返回 None。"""
    gapp = _GRADIO_APP
    if gapp is None:
        return None
    cid = getattr(gapp, "cookie_id", None)
    if not cid:
        return None
    token = request.cookies.get(f"access-token-{cid}") or request.cookies.get(
        f"access-token-unsecure-{cid}"
    )
    if token and token in getattr(gapp, "tokens", {}):
        return gapp.tokens[token]
    return None


def _rebuild_html_response_headers(raw_headers):
    """保留全部原始头（含多个 Set-Cookie），仅排除 content-length。"""
    return [
        (k.decode("latin-1"), v.decode("latin-1"))
        for k, v in raw_headers
        if k.lower() != b"content-length"
    ]


@app.middleware("http")
async def inject_login_css(request: Request, call_next):
    response = await call_next(request)
    if "text/html" not in response.headers.get("content-type", ""):
        return response

    body = b""
    async for chunk in response.body_iterator:
        body += chunk
    if not body:
        return response

    has_login_form = (
        b'name="username"' in body
        and b'name="password"' in body
        and b'type="password"' in body
    )
    is_login_page = (
        request.url.path == "/"
        and _gradio_request_user(request) is None
        and (has_login_form or not request.headers.get("authorization"))
    )

    if is_login_page:
        css_bytes = build_auth_page_css(center_body=has_login_form).encode("utf-8")
        entry_class = (
            "register-entry" if has_login_form else "register-entry register-entry-floating"
        )
        link_bytes = (
            f'<div class="{entry_class}">还没有账号？<a href="/register">立即注册</a></div>'
        ).encode("utf-8")
        if b"</head>" in body:
            body = body.replace(b"</head>", css_bytes + b"</head>")
        elif b"<body" in body:
            body = re.sub(b"(<body[^>]*>)", css_bytes + b"\\1", body)
        if b"</body>" in body:
            body = body.replace(b"</body>", link_bytes + b"</body>")
        else:
            body += link_bytes

    # 管理员只保留管理面板：隐藏四个功能标签页
    username = _gradio_request_user(request)
    if username and request.url.path == "/":
        user = get_user(username)
        if user:
            role_css = _ADMIN_UI_CSS if user.get("role") == "admin" else _USER_UI_CSS
            role_script = (
                _ADMIN_UI_SCRIPT if user.get("role") == "admin" else _USER_UI_SCRIPT
            )
            role_css_bytes = role_css.encode("utf-8")
            role_script_bytes = role_script.encode("utf-8")
            if b"</head>" in body:
                body = body.replace(
                    b"</head>", role_css_bytes + role_script_bytes + b"</head>"
                )
            elif b"<body" in body:
                body = re.sub(
                    b"(<body[^>]*>)", role_css_bytes + role_script_bytes + b"\\1", body
                )

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


# ============================== Gradio UI ==============================
custom_css = """
.start-button { color: blue; margin: 4px 2px; }
.clear-button { color: red; margin: 4px 2px; }
"""

title_text = f"<center><h1>{APP_TITLE}</h1></center>"


def _run_pipeline_wrapper(input_image, user_state, mode):
    """把流水线异常统一转成 Gradio 用户提示。"""
    try:
        res_img, evaluate_text = run_pipeline(input_image, user_state, mode)
        gr.Info("success!")
        if mode != "detect" and evaluate_text and "⚠" in evaluate_text:
            gr.Warning(evaluate_text.splitlines()[-1])
        return res_img, evaluate_text
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"处理失败：{exc}") from exc


def process_image_1(input_image, user_state):
    return _run_pipeline_wrapper(input_image, user_state, "restore")


def process_image_2(input_image, user_state):
    return _run_pipeline_wrapper(input_image, user_state, "restore_scratch")


def process_image_3(input_image, user_state):
    res_img, _ = _run_pipeline_wrapper(input_image, user_state, "detect")
    return res_img


def process_colorize(input_image, user_state):
    """老照片上色（DDColor）。"""
    try:
        res_img = run_colorize(input_image, user_state)
        gr.Info("上色完成!")
        return res_img
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"上色失败：{exc}") from exc


def clear_inputs():
    return None, None, ""


def clear_inputs_3():
    return None, None


def on_load(request: gr.Request):
    """页面加载时设置会话状态（标签页显隐由服务端注入脚本处理）。"""
    username = request.username
    user = get_user(username) if username else None
    role = user.get("role", "user") if user else "user"
    return {"username": username, "role": role}


with gr.Blocks(title=APP_TITLE) as demo:
    user_state = gr.State()
    gr.Markdown(title_text)

    gr.HTML(
        """
        <div style="text-align:right;margin-bottom:10px;">
            <a href="/logout" style="display:inline-block;padding:6px 18px;
                background:#e74c3c;color:#fff;text-decoration:none;
                border-radius:6px;font-size:14px;">
                退出登录 / 切换账户
            </a>
        </div>
        """
    )

    with gr.Tabs():
        with gr.TabItem("不带划痕的旧照片复原模块", elem_id="tab_restore") as tab1:
            with gr.Row():
                with gr.Column():
                    input_image_1 = gr.Image(label="待修复图像")
                    with gr.Row():
                        clear_button_1 = gr.Button("清空图片输入", elem_classes="clear-button")
                        submit_button_1 = gr.Button("开始修复", elem_classes="start-button")
                with gr.Column():
                    result_1 = gr.Image(label="修复结果")
                    elavuation_logs_1 = gr.Textbox(
                        label="修复图与原(退化)图的差异指标(PSNR/SSIM/MAE)", lines=3
                    )
            gr.Examples(
                examples=[
                    ["./gradio_examples/old/a.png"],
                    ["./gradio_examples/old/old_a.png"],
                    ["./gradio_examples/old/b.png"],
                    ["./gradio_examples/old/old_f.png"],
                    ["./gradio_examples/old/old_g.png"],
                    ["./gradio_examples/old/old_h.png"],
                    ["./gradio_examples/old/old_i.png"],
                    ["./gradio_examples/old/old_b.png"],
                    ["./gradio_examples/old/old_c.png"],
                    ["./gradio_examples/old/d.png"],
                    ["./gradio_examples/old/old_d.png"],
                    ["./gradio_examples/old/e.png"],
                    ["./gradio_examples/old/old_e.png"],
                    ["./gradio_examples/old/f.png"],
                    ["./gradio_examples/old/c.png"],
                ],
                inputs=[input_image_1],
                label="以下是一些示例输入,点击即可自动装载到上方"
                "(样例存在第二页,在下方'Pages'处可选择展开第二页样例)",
            )
            submit_button_1.click(
                process_image_1,
                inputs=[input_image_1, user_state],
                outputs=[result_1, elavuation_logs_1],
            )
            clear_button_1.click(
                clear_inputs,
                inputs=[],
                outputs=[input_image_1, result_1, elavuation_logs_1],
            )

        with gr.TabItem("带划痕的旧照片复原模块", elem_id="tab_scratch") as tab2:
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="待修复图像")
                    with gr.Row():
                        clear_button_2 = gr.Button("清空所有内容", elem_classes="clear-button")
                        submit_button_2 = gr.Button("提交复原", elem_classes="start-button")
                with gr.Column():
                    result_2 = gr.Image(label="修复结果")
                    elavuation_logs_2 = gr.Textbox(
                        label="修复图与原(退化)图的差异指标(PSNR/SSIM/MAE)", lines=3
                    )
            gr.Examples(
                examples=[
                    ["./gradio_examples/old_w_scratch/a.png"],
                    ["./gradio_examples/old_w_scratch/b.png"],
                    ["./gradio_examples/old_w_scratch/c.png"],
                    ["./gradio_examples/old_w_scratch/d.png"],
                ],
                inputs=[input_image],
                label="以下是一些示例输入,点击即可自动装载到上方",
            )
            submit_button_2.click(
                process_image_2,
                inputs=[input_image, user_state],
                outputs=[result_2, elavuation_logs_2],
            )
            clear_button_2.click(
                clear_inputs,
                inputs=[],
                outputs=[input_image, result_2, elavuation_logs_2],
            )

        with gr.TabItem("划痕检测模块", elem_id="tab_detect") as tab3:
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="待检测图像")
                    with gr.Row():
                        clear_button_3 = gr.Button("清空所有内容", elem_classes="clear-button")
                        submit_button_3 = gr.Button("提交复原", elem_classes="start-button")
                with gr.Column():
                    result_3 = gr.Image(label="检测结果")
            gr.Examples(
                examples=[
                    ["./gradio_examples/old_w_scratch/a.png"],
                    ["./gradio_examples/old_w_scratch/b.png"],
                    ["./gradio_examples/old_w_scratch/c.png"],
                    ["./gradio_examples/old_w_scratch/d.png"],
                ],
                inputs=[input_image],
                label="以下是一些示例输入,点击即可自动装载到上方",
            )
            submit_button_3.click(
                process_image_3,
                inputs=[input_image, user_state],
                outputs=[result_3],
            )
            clear_button_3.click(
                clear_inputs_3, inputs=[], outputs=[input_image, result_3]
            )

        with gr.TabItem("老照片上色模块", elem_id="tab_colorize") as tab4:
            with gr.Row():
                with gr.Column():
                    input_image_4 = gr.Image(label="待上色图像（黑白/灰度）")
                    with gr.Row():
                        clear_button_4 = gr.Button("清空所有内容", elem_classes="clear-button")
                        submit_button_4 = gr.Button("开始上色", elem_classes="start-button")
                with gr.Column():
                    result_4 = gr.Image(label="上色结果")
            gr.Examples(
                examples=[
                    ["./gradio_examples/color/o1.jpg"],
                    ["./gradio_examples/color/o2.jpg"],
                ],
                inputs=[input_image_4],
                label="以下是一些示例输入,点击即可自动装载到上方",
            )
            submit_button_4.click(
                process_colorize,
                inputs=[input_image_4, user_state],
                outputs=[result_4],
            )
            clear_button_4.click(
                clear_inputs_3, inputs=[], outputs=[input_image_4, result_4]
            )

        with gr.TabItem("管理面板", elem_id="admin_panel") as admin_tab:
            from admin_panel import build_admin_panel

            build_admin_panel(user_state)

    demo.load(on_load, inputs=[], outputs=[user_state])


app = gr.mount_gradio_app(
    app,
    demo,
    path="/",
    auth=auth_fn,
    auth_message="请输入您的用户名和密码",
    max_file_size="10mb",
    show_error=False,
    css=custom_css,
)

# Gradio 以 Mount 形式挂在 "/"（最后一条路由），用于读取会话 token / cookie_id。
_GRADIO_MOUNT = app.routes[-1] if app.routes else None
_GRADIO_APP = getattr(_GRADIO_MOUNT, "app", None) if _GRADIO_MOUNT is not None else None


def _resolve_bind():
    host = os.environ.get("FIXIMG_HOST", "127.0.0.1")
    port = int(os.environ.get("FIXIMG_PORT", "9502"))
    return host, port


if __name__ == "__main__":
    try:
        n = verify_weights()
        print(f"[自检] 权重完整性校验通过（{n} 个文件）")
    except WeightsIntegrityError as exc:
        print(f"[自检] 权重校验失败，服务拒绝启动：\n{exc}", file=sys.stderr)
        sys.exit(1)

    init_db()
    host, port = _resolve_bind()
    print(f"启动服务: http://{host}:{port}")
    uvicorn.run(app="main:app", host=host, port=port, reload=False)
