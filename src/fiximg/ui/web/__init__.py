"""The native web UI: session, pages, routes and the static client.

Replaces the Gradio shell (`fiximg/ui/gradio_app.py` and its panels), which was
removed in the same change because it was the only reason `gradio` appeared in the
published dependency graph — and `gradio` pins `pillow<13`, which is the advisory
this release has to clear. See `reports/p17_fastapi_ui_migration.md`.
"""
