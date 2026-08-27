"""Admin panel (Gradio 6 + SQLite).

All admin callbacks perform server-side role validation (_require_admin) at the function entry,
without relying on frontend Tab hiding; the role is read live from the database.
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

PERMISSION_DENIED_MSG = "⛔ Permission denied: admin only"


def _require_admin(caller_state):
    """Server-side admin role validation: username from session state, role read live from the database."""
    username = (caller_state or {}).get("username")
    if not username:
        return False
    user = get_user(username)
    return bool(user and user.get("role") == "admin")


def refresh_history(caller_state):
    if not _require_admin(caller_state):
        return [["⛔ Permission denied, contact the administrator", "", "", "", "", "", ""]]
    history = list_history(50)
    if not history:
        return [["No records", "", "", "", "", "", ""]]
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
    return gr.update(value="✅ Records cleared", visible=True)


def build_task_management(user_state):
    gr.Markdown("### 📋 Task Processing History")
    refresh_btn = gr.Button("Refresh")
    history_df = gr.Dataframe(
        headers=["Time", "User", "Task Type", "PSNR", "SSIM", "MAE", "Input File"],
        datatype=["str", "str", "str", "str", "str", "str", "str"],
        label="Processing History",
        interactive=False,
    )
    refresh_btn.click(refresh_history, inputs=[user_state], outputs=[history_df])

    gr.Markdown("### 📊 Statistics Overview")
    stats_text = gr.Textbox(label="Statistics", lines=5, interactive=False)
    refresh_btn.click(refresh_stats, inputs=[user_state], outputs=[stats_text])

    with gr.Accordion("Danger Zone: Clear Records", open=False):
        gr.Markdown("⚠️ This will delete all processing history and cannot be undone")
        clear_btn = gr.Button("Clear All Records")
        clear_status = gr.Textbox(label="Result", visible=False)
        clear_btn.click(clear_history_fn, inputs=[user_state], outputs=[clear_status])


def load_archive_list(caller_state):
    if not _require_admin(caller_state):
        return gr.update(choices=[("⛔ Permission denied", "")])
    history = list_history()
    choices = [
        (
            f"[{r.get('timestamp', '')}] {r.get('type', '')} — "
            f"{os.path.basename(r.get('input_path', ''))}",
            r.get("id", ""),
        )
        for r in history
    ]
    return gr.update(choices=choices or [("No records", "")])


def show_archive_entry(caller_state, selected_id):
    if not _require_admin(caller_state):
        return None, None, PERMISSION_DENIED_MSG
    if not selected_id:
        return None, None, "Please select a record"
    record = get_history_record(selected_id)
    if not record:
        return None, None, "Record not found"

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
        f"Time: {record.get('timestamp', '')}\n"
        f"User: {record.get('user', '')}\n"
        f"Task Type: {record.get('type', '')}\n"
        f"PSNR: {record.get('psnr', '')}    SSIM: {record.get('ssim', '')}    "
        f"MAE: {record.get('mae', '')}"
    )
    return in_file, out_file, info


def build_photo_archive(user_state):
    gr.Markdown("### 🖼️ Photo Archive")
    refresh_btn = gr.Button("Refresh")
    history_selector = gr.Dropdown(label="Select a record", choices=[], interactive=True)
    with gr.Row():
        input_img = gr.Image(label="Original Image")
        output_img = gr.Image(label="Restored Image")
    info_text = gr.Textbox(label="Processing Info", lines=5, interactive=False)
    refresh_btn.click(load_archive_list, inputs=[user_state], outputs=[history_selector])
    history_selector.change(
        show_archive_entry,
        inputs=[user_state, history_selector],
        outputs=[input_img, output_img, info_text],
    )


def refresh_users(caller_state):
    if not _require_admin(caller_state):
        return [["⛔ Permission denied", ""]]
    return [[u["username"], u["role"]] for u in list_users()]


def update_dropdowns(caller_state):
    if not _require_admin(caller_state):
        empty = gr.update(choices=[("⛔ Permission denied", "")])
        return empty, empty
    choices = [u["username"] for u in list_users()]
    return gr.update(choices=choices), gr.update(choices=choices)


def do_add_user(caller_state, username, password, role):
    if not _require_admin(caller_state):
        return gr.update(value=PERMISSION_DENIED_MSG, visible=True)
    if not username or not password:
        return gr.update(value="❌ Username and password cannot be empty", visible=True)
    if get_user(username):
        return gr.update(value=f"❌ User '{username}' already exists", visible=True)
    add_user(username, hash_password(password), role)
    return gr.update(value=f"✅ User '{username}' added", visible=True)


def do_edit_user(caller_state, username, password, role):
    if not _require_admin(caller_state):
        return gr.update(value=PERMISSION_DENIED_MSG, visible=True)
    if not username:
        return gr.update(value="❌ Please select a user", visible=True)
    if not get_user(username):
        return gr.update(value=f"❌ User '{username}' does not exist", visible=True)
    update_user(
        username,
        password_hash=hash_password(password) if password else None,
        role=role or None,
    )
    return gr.update(value=f"✅ User '{username}' updated", visible=True)


def do_delete_user(caller_state, username):
    if not _require_admin(caller_state):
        return gr.update(value=PERMISSION_DENIED_MSG, visible=True)
    if not username:
        return gr.update(value="❌ Please select a user", visible=True)
    if username == "admin":
        return gr.update(value="❌ Cannot delete the built-in admin account", visible=True)
    delete_user(username)
    return gr.update(value=f"✅ User '{username}' deleted", visible=True)


def build_user_management(user_state):
    gr.Markdown("### 👥 User Management")
    users_df = gr.Dataframe(
        headers=["Username", "Role"],
        datatype=["str", "str"],
        label="Current Users",
        interactive=False,
    )
    refresh_btn = gr.Button("Refresh User List")
    refresh_btn.click(refresh_users, inputs=[user_state], outputs=[users_df])

    gr.Markdown("---")
    gr.Markdown("### Manage Users")
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Add User")
            new_username = gr.Textbox(label="New Username", placeholder="Enter username")
            new_password = gr.Textbox(label="Password", type="password", placeholder="Enter password")
            new_role = gr.Radio(label="Role", choices=["user", "admin"], value="user")
            add_btn = gr.Button("Add User")
            add_status = gr.Textbox(label="Result", interactive=False, visible=False)

        with gr.Column():
            gr.Markdown("#### Change Password / Role")
            edit_username = gr.Dropdown(label="Select user", choices=[], interactive=True)
            edit_password = gr.Textbox(
                label="New password (leave empty to keep)", type="password", placeholder="Enter new password"
            )
            edit_role = gr.Dropdown(
                label="New role (leave empty to keep)", choices=["user", "admin"], interactive=True
            )
            edit_btn = gr.Button("Save Changes")
            edit_status = gr.Textbox(label="Result", interactive=False, visible=False)

        with gr.Column():
            gr.Markdown("#### Delete User")
            del_username = gr.Dropdown(label="Select user", choices=[], interactive=True)
            del_btn = gr.Button("Delete User")
            del_status = gr.Textbox(label="Result", interactive=False, visible=False)

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
        with gr.TabItem("Task Management"):
            build_task_management(user_state)
        with gr.TabItem("Photo Archive"):
            build_photo_archive(user_state)
        with gr.TabItem("User Management"):
            build_user_management(user_state)

