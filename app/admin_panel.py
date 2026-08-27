"""后台管理面板（Gradio 6 + SQLite）。

所有后台回调在函数入口做服务端角色校验（_require_admin），
不依赖前端 Tab 隐藏；角色从数据库实时读取。
"""
import os
from datetime import datetime

import gradio as gr

from config.security import hash_password
from app.db import (
    ARCHIVE_INPUT_DIR,
    ARCHIVE_OUTPUT_DIR,
    add_user,
    clear_history,
    delete_user,
    get_history_record,
    get_user,
    history_stats,
    list_history,
    list_users,
    update_user,
)

PERMISSION_DENIED_MSG = "⛔ 权限不足：仅管理员可执行此操作"


def _require_admin(caller_state):
    """服务端管理员角色校验：用户名取自会话状态，角色实时查库。"""
    username = (caller_state or {}).get("username")
    if not username:
        return False
    user = get_user(username)
    return bool(user and user.get("role") == "admin")


# ===================== 任务管理 =====================
def refresh_history(caller_state):
    if not _require_admin(caller_state):
        return [["⛔ 权限不足，请联系管理员", "", "", "", "", "", ""]]
    history = list_history(50)
    if not history:
        return [["暂无记录", "", "", "", "", "", ""]]
    return [
        [
            r.get("timestamp", ""),
            r.get("user", ""),
            r.get("type", ""),
            str(r.get("psnr", "")),
            str(r.get("ssim", "")),
            str(r.get("mae", "")),
            os.path.basename(r.get("input_path", "")),
        ]
        for r in history
    ]


def refresh_stats(caller_state):
    if not _require_admin(caller_state):
        return PERMISSION_DENIED_MSG
    return history_stats()


def clear_history_fn(caller_state):
    if not _require_admin(caller_state):
        return gr.update(value=PERMISSION_DENIED_MSG, visible=True)
    clear_history()
    return gr.update(value="✅ 记录已清空", visible=True)


def build_task_management(user_state):
    gr.Markdown("### 📋 任务处理记录")
    refresh_btn = gr.Button("刷新记录")
    history_df = gr.Dataframe(
        headers=["时间", "用户", "任务类型", "PSNR", "SSIM", "MAE", "输入文件"],
        datatype=["str", "str", "str", "str", "str", "str", "str"],
        label="处理历史",
        interactive=False,
    )
    refresh_btn.click(refresh_history, inputs=[user_state], outputs=[history_df])

    gr.Markdown("### 📊 统计概览")
    stats_text = gr.Textbox(label="统计信息", lines=5, interactive=False)
    refresh_btn.click(refresh_stats, inputs=[user_state], outputs=[stats_text])

    with gr.Accordion("危险操作：清空记录", open=False):
        gr.Markdown("⚠️ 此操作将删除所有处理历史记录，不可恢复")
        clear_btn = gr.Button("清空所有处理记录")
        clear_status = gr.Textbox(label="操作结果", visible=False)
        clear_btn.click(clear_history_fn, inputs=[user_state], outputs=[clear_status])


# ===================== 照片档案 =====================
def load_archive_list(caller_state):
    if not _require_admin(caller_state):
        return gr.update(choices=[("⛔ 权限不足", "")])
    history = list_history()
    choices = [
        (
            f"[{r.get('timestamp', '')}] {r.get('type', '')} — "
            f"{os.path.basename(r.get('input_path', ''))}",
            r.get("id", ""),
        )
        for r in history
    ]
    return gr.update(choices=choices or [("暂无记录", "")])


def show_archive_entry(caller_state, selected_id):
    if not _require_admin(caller_state):
        return None, None, PERMISSION_DENIED_MSG
    if not selected_id:
        return None, None, "请选择一条处理记录"
    record = get_history_record(selected_id)
    if not record:
        return None, None, "记录未找到"

    prefix = record.get("id", "")
    in_name = os.path.basename(record.get("input_path", ""))
    out_name = os.path.basename(record.get("output_path", ""))

    in_file = os.path.join(ARCHIVE_INPUT_DIR, f"{prefix}_{in_name}")
    if not os.path.exists(in_file):
        in_path = record.get("input_path", "")
        in_file = in_path if in_path and os.path.exists(in_path) else None

    out_file = os.path.join(ARCHIVE_OUTPUT_DIR, f"{prefix}_{out_name}")
    if not os.path.exists(out_file):
        out_path = record.get("output_path", "")
        out_file = out_path if out_path and os.path.exists(out_path) else None

    info = (
        f"时间: {record.get('timestamp', '')}\n"
        f"用户: {record.get('user', '')}\n"
        f"任务类型: {record.get('type', '')}\n"
        f"PSNR: {record.get('psnr', '')}    SSIM: {record.get('ssim', '')}    "
        f"MAE: {record.get('mae', '')}"
    )
    return in_file, out_file, info


def build_photo_archive(user_state):
    gr.Markdown("### 🖼️ 照片档案")
    refresh_btn = gr.Button("刷新档案")
    history_selector = gr.Dropdown(label="选择处理记录", choices=[], interactive=True)
    with gr.Row():
        input_img = gr.Image(label="原始图片")
        output_img = gr.Image(label="修复后图片")
    info_text = gr.Textbox(label="处理信息", lines=5, interactive=False)
    refresh_btn.click(load_archive_list, inputs=[user_state], outputs=[history_selector])
    history_selector.change(
        show_archive_entry,
        inputs=[user_state, history_selector],
        outputs=[input_img, output_img, info_text],
    )


# ===================== 用户管理 =====================
def refresh_users(caller_state):
    if not _require_admin(caller_state):
        return [["⛔ 权限不足", ""]]
    return [[u["username"], u["role"]] for u in list_users()]


def update_dropdowns(caller_state):
    if not _require_admin(caller_state):
        empty = gr.update(choices=[("⛔ 权限不足", "")])
        return empty, empty
    choices = [u["username"] for u in list_users()]
    return gr.update(choices=choices), gr.update(choices=choices)


def do_add_user(caller_state, username, password, role):
    if not _require_admin(caller_state):
        return gr.update(value=PERMISSION_DENIED_MSG, visible=True)
    if not username or not password:
        return gr.update(value="❌ 用户名和密码不能为空", visible=True)
    if get_user(username):
        return gr.update(value=f"❌ 用户 '{username}' 已存在", visible=True)
    add_user(username, hash_password(password), role)
    return gr.update(value=f"✅ 用户 '{username}' 添加成功", visible=True)


def do_edit_user(caller_state, username, password, role):
    if not _require_admin(caller_state):
        return gr.update(value=PERMISSION_DENIED_MSG, visible=True)
    if not username:
        return gr.update(value="❌ 请选择用户", visible=True)
    if not get_user(username):
        return gr.update(value=f"❌ 用户 '{username}' 不存在", visible=True)
    update_user(
        username,
        password_hash=hash_password(password) if password else None,
        role=role or None,
    )
    return gr.update(value=f"✅ 用户 '{username}' 修改成功", visible=True)


def do_delete_user(caller_state, username):
    if not _require_admin(caller_state):
        return gr.update(value=PERMISSION_DENIED_MSG, visible=True)
    if not username:
        return gr.update(value="❌ 请选择用户", visible=True)
    if username == "admin":
        return gr.update(value="❌ 不能删除内置 admin 账户", visible=True)
    delete_user(username)
    return gr.update(value=f"✅ 用户 '{username}' 已删除", visible=True)


def build_user_management(user_state):
    gr.Markdown("### 👥 用户管理")
    users_df = gr.Dataframe(
        headers=["用户名", "角色"],
        datatype=["str", "str"],
        label="当前用户列表",
        interactive=False,
    )
    refresh_btn = gr.Button("刷新用户列表")
    refresh_btn.click(refresh_users, inputs=[user_state], outputs=[users_df])

    gr.Markdown("---")
    gr.Markdown("### 操作用户")
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### 添加用户")
            new_username = gr.Textbox(label="新用户名", placeholder="请输入用户名")
            new_password = gr.Textbox(label="密码", type="password", placeholder="请输入密码")
            new_role = gr.Radio(label="角色", choices=["user", "admin"], value="user")
            add_btn = gr.Button("添加用户")
            add_status = gr.Textbox(label="操作结果", interactive=False, visible=False)

        with gr.Column():
            gr.Markdown("#### 修改密码 / 角色")
            edit_username = gr.Dropdown(label="选择用户", choices=[], interactive=True)
            edit_password = gr.Textbox(
                label="新密码（留空则不修改）", type="password", placeholder="输入新密码"
            )
            edit_role = gr.Dropdown(
                label="新角色（不选则不修改）", choices=["user", "admin"], interactive=True
            )
            edit_btn = gr.Button("保存修改")
            edit_status = gr.Textbox(label="操作结果", interactive=False, visible=False)

        with gr.Column():
            gr.Markdown("#### 删除用户")
            del_username = gr.Dropdown(label="选择用户", choices=[], interactive=True)
            del_btn = gr.Button("删除用户")
            del_status = gr.Textbox(label="操作结果", interactive=False, visible=False)

    refresh_btn.click(
        update_dropdowns, inputs=[user_state], outputs=[edit_username, del_username]
    )
    add_btn.click(
        do_add_user,
        inputs=[user_state, new_username, new_password, new_role],
        outputs=[add_status],
    )
    edit_btn.click(
        do_edit_user,
        inputs=[user_state, edit_username, edit_password, edit_role],
        outputs=[edit_status],
    )
    del_btn.click(
        do_delete_user, inputs=[user_state, del_username], outputs=[del_status]
    )


def build_admin_panel(user_state):
    with gr.Tabs():
        with gr.TabItem("任务管理"):
            build_task_management(user_state)
        with gr.TabItem("照片档案"):
            build_photo_archive(user_state)
        with gr.TabItem("用户管理"):
            build_user_management(user_state)

