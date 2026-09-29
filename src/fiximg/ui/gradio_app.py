"""Gradio UI (plan §23, §26, §3.9).

Callbacks never touch pipeline internals: every button enqueues a task via the
TaskService and streams stage-level progress (plan §23/§24) through the
generator in ``fiximg/ui/task_progress.py``. If no queue consumer is reachable
the handler falls back to the synchronous path, preserving the V1 UX.

V3 layout (plan §3.9.1) adds two things users asked for:

* a **Before/After slider** next to every result, so the restoration can be
  judged in place instead of comparing two thumbnails side by side;
* a **History** tab that lists past tasks and can re-run one.
"""
import gradio as gr

from fiximg.ui.components import MASK_LABEL, result_column
from fiximg.ui.history_panel import build_history_panel
from fiximg.ui.state import resolve_user_state, tab_visibility
from fiximg.ui.task_progress import (
    build_preview,
    build_switches,
    make_preview_handler,
    make_submit_handler,
)

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
# role stylesheet in fiximg/ui/middleware.py targets.
TAB_ID_AUTO = "auto"
TAB_ID_RESTORE = "restore"
TAB_ID_SCRATCH = "scratch"
TAB_ID_DETECT = "detect"
TAB_ID_COLORIZE = "colorize"
TAB_ID_HISTORY = "history"
TAB_ID_ADMIN = "admin"

#: Which tab ids an administrator sees that an ordinary user does not. **Not** the
#: rule: production's single source for that is `ui/state.py::tab_visibility`.
#: This list is the fixture
#: `test_the_admin_only_tab_list_is_the_one_visibility_enforces` compares the two
#: against, so a third copy of the rule cannot appear without a test noticing.
#: It lives next to the ids it names because the ids are the fact being asserted.
ADMIN_ONLY_TABS = (TAB_ID_ADMIN,)


process_image_1 = make_submit_handler("restore")
process_image_2 = make_submit_handler("restore_scratch")
process_image_3 = make_submit_handler("detect")
process_colorize = make_submit_handler("colorize")
process_auto = make_submit_handler("auto")  # §10 Phase 7 one-click entry


#: Shared by both clear callbacks: the download control has no result to offer once
#: the tab is cleared, so it goes back to hidden rather than keeping a stale path.
_CLEAR_DOWNLOAD = (None, None, "", None, gr.update(visible=False))


def clear_inputs():
    """Clear input + result + metrics + Before/After + download (5 outputs)."""
    return _CLEAR_DOWNLOAD


def clear_inputs_3():
    """Clear input + result + status + Before/After + download (5 outputs)."""
    return _CLEAR_DOWNLOAD


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
      * the role stylesheet injected by fiximg/ui/middleware.py hides those
        buttons (`#{elem_id}-button`), and CSS survives client re-renders.
      * `selected` lands the caller on a tab that is actually visible, instead
        of the default first tab (which is hidden for admins).
    """
    from fiximg.ui.admin_panel import security_banner_text as _banner

    # §3.2: one module owns "who is this caller" (ui/state.py).
    user_state = resolve_user_state(request)
    is_admin = user_state.is_admin
    state = user_state.to_dict()
    visibility = tab_visibility(user_state)
    show_function_tabs = visibility["function_tabs"]

    try:
        banner = _banner(state)
    except Exception:  # noqa: BLE001 — the banner is cosmetic; never block the layout
        banner = ""

    return (
        state,
        # Each TabItem gets its own update object (never a shared one).
        gr.update(visible=show_function_tabs),  # tab_auto
        gr.update(visible=show_function_tabs),  # tab_restore
        gr.update(visible=show_function_tabs),  # tab_scratch
        gr.update(visible=show_function_tabs),  # tab_detect
        gr.update(visible=show_function_tabs),  # tab_colorize
        gr.update(visible=visibility["history_tab"]),  # tab_history
        gr.update(visible=visibility["admin_panel"]),  # admin_panel
        gr.update(selected=TAB_ID_ADMIN if is_admin else TAB_ID_AUTO),
        banner,
    )


#: Result image + Before/After slider, shared by every tab (plan §3.9.2).
_result_column = result_column


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
                        # §3.9.1's 自动分析 row, above the switches: the numbers the Auto
                        # tab's captions refer to, shown before the GPU work starts.
                        preview_auto = build_preview("auto")
                        switches_auto = build_switches("auto")
                        with gr.Row():
                            clear_button_auto = gr.Button("Clear All", elem_classes="clear-button")
                            submit_button_auto = gr.Button("Start Auto Restore", elem_classes="start-button")
                    with gr.Column():
                        result_auto, compare_auto, download_auto = _result_column(
                            "Restored Result", "compare_auto"
                        )
                        evaluation_logs_auto = gr.Textbox(
                            label="Evaluation & automatic pipeline decisions", lines=3
                        )
                gr.Examples(
                    # Every example in the repository. This tab's plan is derived
                    # from the picture, so it accepts anything the other three
                    # accept — clean, scratched and grey alike — and trimming the
                    # set to three removed the only place a new arrival could see
                    # what the analyser does with a photo they recognise.
                    # `tests/ui/test_examples_all_reachable.py` fails if a file is
                    # added to `examples/` and not offered here or on another tab.
                    examples=[[f"./examples/old/{name}.png"] for name in "abcdefgh"]
                             + [[f"./examples/old_w_scratch/{name}.png"] for name in "abcd"]
                             + [[f"./examples/color/{name}.jpg"] for name in ("o1", "o2")],
                    inputs=[input_image_auto],
                    label="Click an example to load it above"
                          "(samples are on a second page; expand it via 'Pages' below)",
                )
                submit_button_auto.click(
                    process_auto,
                    inputs=[input_image_auto, user_state, *switches_auto],
                    outputs=[result_auto, evaluation_logs_auto, compare_auto, download_auto],
                    concurrency_limit=1,
                )
                # The row re-renders on the upload *and* on every switch, because the
                # switches are part of the answer: untick "face enhancement" and the
                # planned pipeline has to lose the chain there and then.
                preview_inputs = [input_image_auto, *switches_auto]
                preview_handler = make_preview_handler("auto")
                input_image_auto.change(
                    preview_handler, inputs=preview_inputs, outputs=[preview_auto]
                )
                for auto_switch in switches_auto:
                    auto_switch.change(
                        preview_handler, inputs=preview_inputs, outputs=[preview_auto]
                    )
                clear_button_auto.click(
                    clear_inputs, inputs=[],
                    outputs=[input_image_auto, result_auto, evaluation_logs_auto, compare_auto, download_auto],
                )

            with gr.TabItem("Restore Old Photo (No Scratches)", elem_id="tab_restore", id=TAB_ID_RESTORE) as tab_restore:
                with gr.Row():
                    with gr.Column():
                        input_image_1 = gr.Image(label="Image to Restore")
                        switches_1 = build_switches("restore")
                        with gr.Row():
                            clear_button_1 = gr.Button("Clear Image", elem_classes="clear-button")
                            submit_button_1 = gr.Button("Start Restoration", elem_classes="start-button")
                    with gr.Column():
                        result_1, compare_1, download_1 = _result_column(
                            "Restored Result", "compare_restore"
                        )
                        evaluation_logs_1 = gr.Textbox(
                            label="Difference metrics vs. degraded original (PSNR/SSIM/MAE)", lines=3
                        )
                gr.Examples(
                    examples=[["./examples/old/a.png"], ["./examples/old/b.png"],
                              ["./examples/old/c.png"], ["./examples/old/d.png"],
                              ["./examples/old/e.png"], ["./examples/old/f.png"],
                              ["./examples/old/g.png"], ["./examples/old/h.png"]],
                    inputs=[input_image_1],
                    label="Click an example to load it above"
                          "(samples are on a second page; expand it via 'Pages' below)",
                )
                submit_button_1.click(
                    process_image_1,
                    inputs=[input_image_1, user_state, *switches_1],
                    outputs=[result_1, evaluation_logs_1, compare_1, download_1],
                    concurrency_limit=1,
                )
                clear_button_1.click(
                    clear_inputs, inputs=[],
                    outputs=[input_image_1, result_1, evaluation_logs_1, compare_1, download_1],
                )

            with gr.TabItem("Restore Old Photo (With Scratches)", elem_id="tab_scratch", id=TAB_ID_SCRATCH) as tab_scratch:
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(label="Image to Restore")
                        switches_2 = build_switches("restore_scratch")
                        with gr.Row():
                            clear_button_2 = gr.Button("Clear All", elem_classes="clear-button")
                            submit_button_2 = gr.Button("Start Scratch Repair", elem_classes="start-button")
                    with gr.Column():
                        result_2, compare_2, download_2 = _result_column(
                            "Restored Result", "compare_scratch"
                        )
                        evaluation_logs_2 = gr.Textbox(
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
                    inputs=[input_image, user_state, *switches_2],
                    outputs=[result_2, evaluation_logs_2, compare_2, download_2],
                )
                clear_button_2.click(
                    clear_inputs, inputs=[],
                    outputs=[input_image, result_2, evaluation_logs_2, compare_2, download_2],
                )

            with gr.TabItem("Scratch Detection", elem_id="tab_detect", id=TAB_ID_DETECT) as tab_detect:
                with gr.Row():
                    with gr.Column():
                        input_image_d = gr.Image(label="Image to Detect")
                        with gr.Row():
                            clear_button_3 = gr.Button("Clear All", elem_classes="clear-button")
                            submit_button_3 = gr.Button("Detect Scratches", elem_classes="start-button")
                    with gr.Column():
                        result_3, compare_3, download_3 = _result_column(
                            "Detection Result (scratch mask)",
                            "compare_detect",
                            slider_label=MASK_LABEL,
                        )
                        status_3 = gr.Textbox(
                            label="Detection status", lines=3, interactive=False
                        )
                gr.Examples(
                    examples=[["./examples/old_w_scratch/a.png"], ["./examples/old_w_scratch/b.png"],
                              ["./examples/old_w_scratch/c.png"], ["./examples/old_w_scratch/d.png"]],
                    inputs=[input_image_d],
                    label="Click an example to load it above",
                )
                # NOTE: the handler is a generator yielding
                # (image, status_text, comparison); it must declare exactly as
                # many outputs as it yields or Gradio tries to postprocess the
                # whole tuple as an image and raises ComponentProcessingError.
                submit_button_3.click(
                    process_image_3,
                    inputs=[input_image_d, user_state],
                    outputs=[result_3, status_3, compare_3, download_3],
                )
                clear_button_3.click(
                    clear_inputs_3, inputs=[],
                    outputs=[input_image_d, result_3, status_3, compare_3, download_3],
                )

            with gr.TabItem("Old Photo Colorization", elem_id="tab_colorize", id=TAB_ID_COLORIZE) as tab_colorize:
                with gr.Row():
                    with gr.Column():
                        input_image_4 = gr.Image(label="Image to Colorize (B&W / Grayscale)")
                        with gr.Row():
                            clear_button_4 = gr.Button("Clear All", elem_classes="clear-button")
                            submit_button_4 = gr.Button("Start Colorization", elem_classes="start-button")
                    with gr.Column():
                        result_4, compare_4, download_4 = _result_column(
                            "Colorized Result", "compare_colorize"
                        )
                        status_4 = gr.Textbox(
                            label="Colorization status", lines=3, interactive=False
                        )
                gr.Examples(
                    examples=[["./examples/color/o1.jpg"], ["./examples/color/o2.jpg"]],
                    inputs=[input_image_4],
                    label="Click an example to load it above",
                )
                # Three outputs required — see the note on submit_button_3.
                submit_button_4.click(
                    process_colorize,
                    inputs=[input_image_4, user_state],
                    outputs=[result_4, status_4, compare_4, download_4],
                )
                clear_button_4.click(
                    clear_inputs_3, inputs=[],
                    outputs=[input_image_4, result_4, status_4, compare_4, download_4],
                )

            with gr.TabItem("History", elem_id="tab_history", id=TAB_ID_HISTORY) as tab_history:
                build_history_panel(user_state)

            with gr.TabItem("Admin Panel", elem_id="admin_panel", id=TAB_ID_ADMIN, visible=False) as tab_admin:
                from fiximg.ui.admin_panel import build_admin_panel

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
                tab_history,
                tab_admin,
                tabs,
                security_banner,
            ],
        )
    return demo
