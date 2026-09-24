"""Gradio UI (extracted from V1 main.py; plan sections 23 and 26).

Callbacks never touch pipeline internals: every button enqueues a task via the
TaskService and streams stage-level progress (plan §23/§24) through the
generator in app/ui/task_progress.py. If no queue consumer is reachable the
handler falls back to the synchronous path, preserving the V1 UX.
"""
import gradio as gr

from app.repositories.user_repository import get_user
from app.ui.task_progress import make_submit_handler

APP_TITLE = (
    "Old Photo Restoration & Scratch Repair via Deep Learning "
    "(GANs and Variational Autoencoders)"
)

custom_css = """
.start-button { color: blue; margin: 4px 2px; }
.clear-button { color: red; margin: 4px 2px; }
"""

# Stable tab identifiers. Gradio only honours a `selected` update when the
# TabItem declares an explicit `id` (see gr.TabItem docs), and it attaches
# "{elem_id}-button" to the tab button in the DOM — which is what the
# role stylesheet in app/ui/middleware.py targets.
TAB_ID_AUTO = "auto"
TAB_ID_RESTORE = "restore"
TAB_ID_SCRATCH = "scratch"
TAB_ID_DETECT = "detect"
TAB_ID_COLORIZE = "colorize"
TAB_ID_ADMIN = "admin"


process_image_1 = make_submit_handler("restore")
process_image_2 = make_submit_handler("restore_scratch")
process_image_3 = make_submit_handler("detect")
process_colorize = make_submit_handler("colorize")
process_auto = make_submit_handler("auto")  # §10 Phase 7 one-click entry


def clear_inputs():
    return None, None, ""


def clear_inputs_3():
    """Clear input + result + status (3 outputs: detect and colorize tabs)."""
    return None, None, ""


def apply_role(request: gr.Request):
    """Resolve the caller's role and return per-tab visibility updates.

    Role-based layout is applied *server-side* through the component props:
    Gradio serialises `visible` per session, so the correct tabs survive every
    client re-render. Patching the DOM from injected CSS/JS (the previous
    approach) did not survive re-renders, which is why users still saw the
    Admin Panel tab and admins still saw the restoration tabs.

    Returns (user_state, *visibility_updates, tabs_selected, security_banner_html).

    Two mechanisms are needed, because neither is sufficient alone:
      * `visible` on each TabItem removes the tab *content* server-side, but
        Gradio's Tabs component does not rebuild its button list when `visible`
        changes at runtime — hidden tabs kept their buttons.
      * the role stylesheet injected by app/ui/middleware.py hides those
        buttons (`#{elem_id}-button`), and CSS survives client re-renders.
      * `selected` lands the caller on a tab that is actually visible, instead
        of the default first tab (which is hidden for admins).
    """
    from app.ui.admin_panel import security_banner_text as _banner

    username = request.username
    try:
        user = get_user(username) if username else None
    except Exception:  # noqa: BLE001 — fail safe to the non-admin layout
        user = None
    role = (user or {}).get("role", "user")
    is_admin = role == "admin"
    state = {"username": username, "role": role}

    try:
        banner = _banner(state)
    except Exception:  # noqa: BLE001 — the banner is cosmetic; never block the layout
        banner = ""

    return (
        state,
        # Each TabItem gets its own update object (never a shared one).
        gr.update(visible=not is_admin),  # tab_auto
        gr.update(visible=not is_admin),  # tab_restore
        gr.update(visible=not is_admin),  # tab_scratch
        gr.update(visible=not is_admin),  # tab_detect
        gr.update(visible=not is_admin),  # tab_colorize
        gr.update(visible=is_admin),      # admin_panel
        gr.update(selected=TAB_ID_ADMIN if is_admin else TAB_ID_AUTO),
        banner,
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        user_state = gr.State()
        gr.Markdown(f"<center><h1>{APP_TITLE}</h1></center>")

        gr.HTML(
            """
            <div style="text-align:right;margin-bottom:10px;">
                <a href="/logout" style="display:inline-block;padding:6px 18px;
                    background:#e74c3c;color:#fff;text-decoration:none;
                    border-radius:6px;font-size:14px;">
                    Sign out / Switch account
                </a>
            </div>
            """
        )

        with gr.Tabs() as tabs:
            with gr.TabItem("Auto Restore (Recommended)", elem_id="tab_auto", id=TAB_ID_AUTO) as tab_auto:
                with gr.Row():
                    with gr.Column():
                        input_image_auto = gr.Image(label="Image to Restore (any old photo)")
                        with gr.Row():
                            clear_button_auto = gr.Button("Clear All", elem_classes="clear-button")
                            submit_button_auto = gr.Button("Start Auto Restore", elem_classes="start-button")
                    with gr.Column():
                        result_auto = gr.Image(label="Restored Result")
                        elavuation_logs_auto = gr.Textbox(
                            label="Evaluation & automatic pipeline decisions", lines=3
                        )
                gr.Examples(
                    examples=[["./examples/old/a.png"], ["./examples/old/old_a.png"],
                              ["./examples/old_w_scratch/a.png"]],
                    inputs=[input_image_auto],
                    label="Click an example to load it above",
                )
                submit_button_auto.click(
                    process_auto,
                    inputs=[input_image_auto, user_state],
                    outputs=[result_auto, elavuation_logs_auto],
                    concurrency_limit=1,
                )
                clear_button_auto.click(
                    clear_inputs, inputs=[], outputs=[input_image_auto, result_auto, elavuation_logs_auto]
                )

            with gr.TabItem("Restore Old Photo (No Scratches)", elem_id="tab_restore", id=TAB_ID_RESTORE) as tab_restore:
                with gr.Row():
                    with gr.Column():
                        input_image_1 = gr.Image(label="Image to Restore")
                        with gr.Row():
                            clear_button_1 = gr.Button("Clear Image", elem_classes="clear-button")
                            submit_button_1 = gr.Button("Start Restoration", elem_classes="start-button")
                    with gr.Column():
                        result_1 = gr.Image(label="Restored Result")
                        elavuation_logs_1 = gr.Textbox(
                            label="Difference metrics vs. degraded original (PSNR/SSIM/MAE)", lines=3
                        )
                gr.Examples(
                    examples=[["./examples/old/a.png"], ["./examples/old/old_a.png"],
                              ["./examples/old/b.png"], ["./examples/old/old_f.png"],
                              ["./examples/old/old_g.png"], ["./examples/old/old_h.png"],
                              ["./examples/old/old_i.png"], ["./examples/old/old_b.png"],
                              ["./examples/old/old_c.png"], ["./examples/old/d.png"],
                              ["./examples/old/old_d.png"], ["./examples/old/e.png"],
                              ["./examples/old/old_e.png"], ["./examples/old/f.png"],
                              ["./examples/old/c.png"]],
                    inputs=[input_image_1],
                    label="Click an example to load it above"
                          "(samples are on a second page; expand it via 'Pages' below)",
                )
                submit_button_1.click(
                    process_image_1,
                    inputs=[input_image_1, user_state],
                    outputs=[result_1, elavuation_logs_1],
                    concurrency_limit=1,
                )
                clear_button_1.click(
                    clear_inputs, inputs=[], outputs=[input_image_1, result_1, elavuation_logs_1]
                )

            with gr.TabItem("Restore Old Photo (With Scratches)", elem_id="tab_scratch", id=TAB_ID_SCRATCH) as tab_scratch:
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(label="Image to Restore")
                        with gr.Row():
                            clear_button_2 = gr.Button("Clear All", elem_classes="clear-button")
                            submit_button_2 = gr.Button("Submit Restoration", elem_classes="start-button")
                    with gr.Column():
                        result_2 = gr.Image(label="Restored Result")
                        elavuation_logs_2 = gr.Textbox(
                            label="Difference metrics vs. degraded original (PSNR/SSIM/MAE)", lines=3
                        )
                gr.Examples(
                    examples=[["./examples/old_w_scratch/a.png"], ["./examples/old_w_scratch/b.png"],
                              ["./examples/old_w_scratch/c.png"], ["./examples/old_w_scratch/d.png"]],
                    inputs=[input_image],
                    label="Click an example to load it above",
                )
                submit_button_2.click(
                    process_image_2,
                    inputs=[input_image, user_state],
                    outputs=[result_2, elavuation_logs_2],
                )
                clear_button_2.click(
                    clear_inputs, inputs=[], outputs=[input_image, result_2, elavuation_logs_2]
                )

            with gr.TabItem("Scratch Detection", elem_id="tab_detect", id=TAB_ID_DETECT) as tab_detect:
                with gr.Row():
                    with gr.Column():
                        input_image_d = gr.Image(label="Image to Detect")
                        with gr.Row():
                            clear_button_3 = gr.Button("Clear All", elem_classes="clear-button")
                            submit_button_3 = gr.Button("Submit Restoration", elem_classes="start-button")
                    with gr.Column():
                        result_3 = gr.Image(label="Detection Result")
                        status_3 = gr.Textbox(
                            label="Detection status", lines=3, interactive=False
                        )
                gr.Examples(
                    examples=[["./examples/old_w_scratch/a.png"], ["./examples/old_w_scratch/b.png"],
                              ["./examples/old_w_scratch/c.png"], ["./examples/old_w_scratch/d.png"]],
                    inputs=[input_image_d],
                    label="Click an example to load it above",
                )
                # NOTE: the handler is a generator yielding (image, status_text);
                # it must declare exactly two outputs or Gradio tries to
                # postprocess the whole tuple as an image and raises
                # ComponentProcessingError.
                submit_button_3.click(
                    process_image_3,
                    inputs=[input_image_d, user_state],
                    outputs=[result_3, status_3],
                )
                clear_button_3.click(
                    clear_inputs_3, inputs=[], outputs=[input_image_d, result_3, status_3]
                )

            with gr.TabItem("Old Photo Colorization", elem_id="tab_colorize", id=TAB_ID_COLORIZE) as tab_colorize:
                with gr.Row():
                    with gr.Column():
                        input_image_4 = gr.Image(label="Image to Colorize (B&W / Grayscale)")
                        with gr.Row():
                            clear_button_4 = gr.Button("Clear All", elem_classes="clear-button")
                            submit_button_4 = gr.Button("Start Colorization", elem_classes="start-button")
                    with gr.Column():
                        result_4 = gr.Image(label="Colorized Result")
                        status_4 = gr.Textbox(
                            label="Colorization status", lines=3, interactive=False
                        )
                gr.Examples(
                    examples=[["./examples/color/o1.jpg"], ["./examples/color/o2.jpg"]],
                    inputs=[input_image_4],
                    label="Click an example to load it above",
                )
                # Two outputs required — see the note on submit_button_3.
                submit_button_4.click(
                    process_colorize,
                    inputs=[input_image_4, user_state],
                    outputs=[result_4, status_4],
                )
                clear_button_4.click(
                    clear_inputs_3, inputs=[], outputs=[input_image_4, result_4, status_4]
                )

            with gr.TabItem("Admin Panel", elem_id="admin_panel", id=TAB_ID_ADMIN, visible=False) as tab_admin:
                from app.ui.admin_panel import build_admin_panel

                security_banner = build_admin_panel(user_state)

        # Single page-load event: the caller's role drives both the tab layout
        # and the §21 banner, so state and visibility can never disagree (two
        # separate load events would race each other).
        demo.load(
            apply_role,
            inputs=[],
            outputs=[
                user_state,
                tab_auto,
                tab_restore,
                tab_scratch,
                tab_detect,
                tab_colorize,
                tab_admin,
                tabs,
                security_banner,
            ],
        )
    return demo
