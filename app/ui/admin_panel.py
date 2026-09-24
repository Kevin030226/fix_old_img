"""Admin panel (Gradio 6) — migrated from V1 app.admin_panel to the V2 services layer.

All admin callbacks perform server-side role validation (_require_admin) at the
function entry; the role is read live from the database.
"""
import os

import gradio as gr

from app.services import history_service
from app.services.user_service import (
    create_user,
    delete_user,
    get_user,
    list_users,
    update_user,
)

PERMISSION_DENIED_MSG = "⛔ Permission denied: admin only"


def _require_admin(caller_state):
    """Server-side admin role validation: username from session state, role read live."""
    username = (caller_state or {}).get("username")
    if not username:
        return False
    user = get_user(username)
    return bool(user and user.get("role") == "admin")


def refresh_history(caller_state):
    if not _require_admin(caller_state):
        return [["⛔ Permission denied, contact the administrator", "", "", "", "", "", ""]]
    history = history_service.list_history(50)
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
    return history_service.history_stats()


def clear_history_fn(caller_state):
    if not _require_admin(caller_state):
        return gr.update(value=PERMISSION_DENIED_MSG, visible=True)
    history_service.clear_history()
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
    history = history_service.list_history()
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
    record = history_service.get_history_record(selected_id)
    if not record:
        return None, None, "Record not found"

    prefix = record.get("id", "")
    in_name = os.path.basename(record.get("input_path", ""))
    out_name = os.path.basename(record.get("output_path", ""))

    in_file = os.path.join(history_service.ARCHIVE_INPUT_DIR, f"{prefix}_{in_name}")
    if not os.path.exists(in_file):
        in_path = record.get("input_path", "")
        in_file = in_path if in_path and os.path.exists(in_path) else None

    out_file = os.path.join(history_service.ARCHIVE_OUTPUT_DIR, f"{prefix}_{out_name}")
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
    create_user(username, password, role)
    return gr.update(value=f"✅ User '{username}' added", visible=True)


def do_edit_user(caller_state, username, password, role):
    if not _require_admin(caller_state):
        return gr.update(value=PERMISSION_DENIED_MSG, visible=True)
    if not username:
        return gr.update(value="❌ Please select a user", visible=True)
    if not get_user(username):
        return gr.update(value=f"❌ User '{username}' does not exist", visible=True)
    update_user(username, password=password or None, role=role or None)
    return gr.update(value=f"✅ User '{username}' updated", visible=True)


def do_change_own_password(caller_state, new_password, confirm_password):
    """§21 force-change entry point: replace the caller's own password."""
    username = (caller_state or {}).get("username")
    if not username:
        return gr.update(value="❌ Not signed in", visible=True)
    if not new_password or len(new_password) < 8:
        return gr.update(
            value="❌ New password must be at least 8 characters", visible=True
        )
    if new_password != confirm_password:
        return gr.update(value="❌ The two passwords do not match", visible=True)
    update_user(username, password=new_password)
    return gr.update(
        value=f"✅ Password changed for '{username}'. All features are unlocked.",
        visible=True,
    )


def security_banner_text(caller_state) -> str:
    """§21 force-change warning text; empty when the account is settled."""
    username = (caller_state or {}).get("username") if isinstance(caller_state, dict) else None
    if not username:
        return ""
    try:
        user = get_user(username)
    except Exception:  # noqa: BLE001 — a missing/uninitialised DB must not break page load
        return ""
    if user and user.get("must_change_password"):
        return (
            "<div style='border:1px solid #f97316;background:#431407;"
            "color:#fdba74;padding:12px 14px;border-radius:8px;font-weight:600'>"
            "⚠ Your account still uses the initial one-time password. Set a new "
            "password below — task submission stays locked until you do.</div>"
        )
    return ""


def build_account_security(user_state):
    """§21 force-change UI: the banner is filled in by the page-load handler
    (build time has no session yet, so no user lookup happens here)."""
    banner = gr.Markdown("")
    gr.Markdown("### 🔑 Account / Change Password")
    new_password = gr.Textbox(label="New password", type="password")
    confirm_password = gr.Textbox(label="Confirm new password", type="password")
    change_btn = gr.Button("Change My Password", variant="primary")
    change_status = gr.Textbox(label="Result", interactive=False, visible=False)
    change_btn.click(
        do_change_own_password,
        inputs=[user_state, new_password, confirm_password],
        outputs=[change_status],
    )
    return banner


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
    banner = None
    with gr.Tabs():
        with gr.TabItem("Account"):
            banner = build_account_security(user_state)
        with gr.TabItem("Task Management"):
            build_task_management(user_state)
        with gr.TabItem("Photo Archive"):
            build_photo_archive(user_state)
        with gr.TabItem("User Management"):
            build_user_management(user_state)

    # The caller (gradio_app) wires demo.load -> security_banner_text -> banner
    # so the §21 force-change warning fills in once the session is known.
    return banner
