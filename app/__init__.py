"""Old photo restoration system — application package (V2 layout).

Layering (plan section 27):
    app/api/          FastAPI routers
    app/ui/           Gradio blocks + HTML pages + middleware
    app/services/     task / artifact / evaluation / user / history services
    app/inference/    stages / model manager / planner / orchestrator / worker
    app/repositories/ SQL access (users, tasks)
    app/core/         config / security / logging / exceptions
    app/schemas/      shared dataclasses
"""
