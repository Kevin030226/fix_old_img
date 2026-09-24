"""Auth router: registration page + rate-limited sign-up (V1 behavior preserved)."""
import re

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from urllib.parse import parse_qs

from app.services import user_service
from app.ui.auth_pages import render_register_page
from config.ratelimit import (
    client_ip,
    register_global_limiter,
    register_ip_limiter,
    register_username_limiter,
)

router = APIRouter(tags=["auth"])


@router.get("/register", response_class=HTMLResponse)
async def register_page():
    return HTMLResponse(render_register_page())


@router.post("/register", response_class=HTMLResponse)
async def register_user(request: Request):
    ip = client_ip(request)
    if not register_ip_limiter.hit(ip):
        return HTMLResponse(
            render_register_page(
                "Too many registration attempts, please try again later "
                "(contact the administrator if the issue persists)"
            ),
            status_code=429,
        )
    if not register_global_limiter.hit("__global__"):
        return HTMLResponse(
            render_register_page("Too many registrations at the moment, please try again later"),
            status_code=429,
        )

    body = (await request.body()).decode("utf-8")
    form = parse_qs(body)
    username = form.get("username", [""])[0].strip()
    password = form.get("password", [""])[0]
    confirm_password = form.get("confirm_password", [""])[0]

    if username and not register_username_limiter.hit(username):
        return HTMLResponse(
            render_register_page(
                "Too many attempts for this username, please try again later", username=username
            ),
            status_code=429,
        )
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,32}", username):
        return HTMLResponse(render_register_page("Invalid username format", username=username))
    if len(password) < 6:
        return HTMLResponse(
            render_register_page("Password must be at least 6 characters", username=username)
        )
    if password != confirm_password:
        return HTMLResponse(
            render_register_page("The two passwords do not match", username=username)
        )
    if not user_service.register_user(username, password):
        return HTMLResponse(
            render_register_page(f"User '{username}' already exists", username=username)
        )
    return HTMLResponse(
        render_register_page(
            "Registration succeeded, please sign in on the login page",
            success=True,
            username=username,
        )
    )
