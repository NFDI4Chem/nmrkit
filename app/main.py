import os
from fastapi import FastAPI, status
from fastapi.responses import RedirectResponse
from fastapi_versioning import VersionedFastAPI

from .routers import registration
from .routers import chem
from .routers import spectra, converter
from fastapi.middleware.cors import CORSMiddleware

from app.core import config, tasks

from prometheus_fastapi_instrumentator import Instrumentator
from app.schemas import HealthCheck

app = FastAPI(
    title=config.PROJECT_NAME,
    description="Python-based microservice to store and predict spectra.",
    terms_of_service="https://nfdi4chem.github.io/nmrkit",
    contact={
        "name": "Steinbeck Lab",
        "url": "https://cheminf.uni-jena.de/",
        "email": "caffeine@listserv.uni-jena.de",
    },
    license_info={
        "name": "CC BY 4.0",
        "url": "https://creativecommons.org/licenses/by/4.0/",
    },
)

app.include_router(registration.router)
app.include_router(chem.router)
app.include_router(spectra.router)
app.include_router(converter.router)

app.add_event_handler("startup", tasks.create_start_app_handler(app))
app.add_event_handler("shutdown", tasks.create_stop_app_handler(app))

app = VersionedFastAPI(
    app,
    version_format="{major}",
    prefix_format="/v{major}",
    enable_latest=True,
    terms_of_service="https://nfdi4chem.github.io/nmrkit",
    contact={
        "name": "Steinbeck Lab",
        "url": "https://cheminf.uni-jena.de/",
        "email": "caffeine@listserv.uni-jena.de",
    },
    license_info={
        "name": "CC BY 4.0",
        "url": "https://creativecommons.org/licenses/by/4.0/",
    },
)

Instrumentator().instrument(app).expose(app)

origins = ["*"]


@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url=os.getenv("HOMEPAGE_URL", "/latest/docs"))


@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    """
    ## Perform a Health Check
    Endpoint to perform a health check on. This endpoint can primarily be used by Docker
    to ensure a robust container orchestration and management are in place. Other
    services that rely on the proper functioning of the API service will not deploy if this
    endpoint returns any other HTTP status code except 200 (OK).
    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    return HealthCheck(status="OK")
