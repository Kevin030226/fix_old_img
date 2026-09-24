"""Pipeline mode labels shared between the Gradio UI and the task service."""

PIPELINE_MODES = {
    "restore": {"label": "Restore (no scratches)", "task_type": "restore"},
    "restore_scratch": {"label": "Restore (with scratches)", "task_type": "restore_scratch"},
    "detect": {"label": "Scratch detection", "task_type": "detect_scratch"},
    "colorize": {"label": "Photo colorization", "task_type": "colorize"},
    # §10/§11 Phase 7: one-click entry; the analyzer derives the pipeline.
    "auto": {"label": "Auto Restore (analyze + repair)", "task_type": "auto_restore"},
}
