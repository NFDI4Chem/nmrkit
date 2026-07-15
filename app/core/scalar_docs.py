from fastapi import FastAPI
from fastapi.routing import APIRoute
from scalar_fastapi import get_scalar_api_reference
from starlette.routing import Mount


def _remove_doc_routes(fastapi_app: FastAPI) -> None:
    fastapi_app.router.routes = [
        route
        for route in fastapi_app.router.routes
        if not (
            isinstance(route, APIRoute)
            and route.path in ("/docs", "/redoc")
        )
    ]
    fastapi_app.docs_url = None
    fastapi_app.redoc_url = None


def _remove_versioning_noop_routes(parent_app: FastAPI) -> None:
    parent_app.router.routes = [
        route
        for route in parent_app.router.routes
        if not (
            isinstance(route, APIRoute)
            and route.path.endswith(("/docs", "/openapi.json"))
            and route.endpoint.__name__ == "noop"
        )
    ]


def _add_scalar_docs(fastapi_app: FastAPI, openapi_url: str, title: str) -> None:
    _remove_doc_routes(fastapi_app)

    @fastapi_app.get("/docs", include_in_schema=False)
    async def scalar_html():
        return get_scalar_api_reference(
            openapi_url=openapi_url,
            title=title,
        )


def configure_scalar_docs(parent_app: FastAPI) -> None:
    """Replace default Swagger UI with Scalar on all versioned API mounts."""
    _remove_versioning_noop_routes(parent_app)

    for route in parent_app.routes:
        if not isinstance(route, Mount) or not isinstance(route.app, FastAPI):
            continue

        prefix = route.path.rstrip("/")
        mounted_app = route.app
        _add_scalar_docs(
            mounted_app,
            openapi_url=f"{prefix}/openapi.json",
            title=mounted_app.title,
        )
