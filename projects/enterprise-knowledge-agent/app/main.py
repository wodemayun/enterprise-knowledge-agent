import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.dependencies import AppContainer, build_container
from app.api.routes import router
from app.core.errors import DomainError


def create_app(container: AppContainer | None = None) -> FastAPI:
    app = FastAPI(title="Enterprise Knowledge Agent", version="0.1.0")
    app.state.container = container or build_container()
    app.include_router(router)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(DomainError)
    async def domain_error_handler(_request: Request, error: DomainError):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    return app


app = create_app()
