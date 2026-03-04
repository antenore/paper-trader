"""Simple CSRF protection using the cookie double-submit pattern.

A random token is set as an HTTP-only cookie. Templates include the same token
as a hidden form field.  On POST, the middleware verifies the two match.

Note: We cache the parsed form data in request.state so that route handlers
can access it after the middleware has consumed the body stream.
"""

from __future__ import annotations

import secrets

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

COOKIE_NAME = "csrf_token"
FORM_FIELD = "csrf_token"
TOKEN_LENGTH = 32


def _get_or_set_token(request: Request, response: Response) -> str:
    """Return the existing CSRF token or generate a new one."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        token = secrets.token_urlsafe(TOKEN_LENGTH)
        response.set_cookie(
            COOKIE_NAME,
            token,
            httponly=True,
            samesite="strict",
            secure=False,  # local dashboard, no TLS
        )
    return token


class CSRFMiddleware(BaseHTTPMiddleware):
    """Verify CSRF token on state-changing requests.

    BaseHTTPMiddleware wraps the request, so form data parsed here is NOT
    available to the route handler via request.form() (body stream consumed).
    We store the parsed form in request.state._csrf_form so route handlers
    can fall back to it.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            cookie_token = request.cookies.get(COOKIE_NAME)
            if cookie_token:
                # Check form data for the token
                content_type = request.headers.get("content-type", "")
                if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
                    form = await request.form()
                    # Cache for downstream handlers (body stream is consumed)
                    request.state._csrf_form = form
                    form_token = form.get(FORM_FIELD)
                    if not form_token or form_token != cookie_token:
                        return Response("CSRF token mismatch", status_code=403)
                # JSON API calls (like /trigger/) are exempt — they use fetch()
                # and are protected by same-origin policy

        response = await call_next(request)

        # Ensure a CSRF cookie is always set
        if not request.cookies.get(COOKIE_NAME):
            token = secrets.token_urlsafe(TOKEN_LENGTH)
            response.set_cookie(
                COOKIE_NAME,
                token,
                httponly=True,
                samesite="strict",
                secure=False,
            )

        return response


def csrf_token(request: Request) -> str:
    """Get the current CSRF token from the request cookie.

    Use in templates: {{ csrf_token }}
    """
    return request.cookies.get(COOKIE_NAME, "")
