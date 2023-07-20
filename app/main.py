from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi_versioning import VersionedFastAPI

from .routers import chem
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
import os

from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="NMR Predict Microservice",
    description="Python based microservice to store and predict spectra.",
    terms_of_service="https://nfdi4chem.github.io/nmr-predict",
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

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chem.router)

app = VersionedFastAPI(
    app,
    version_format="{major}",
    prefix_format="/v{major}",
    enable_latest=True,
    terms_of_service="https://nfdi4chem.github.io/nmr-predict",
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


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(
        url="https://nfdi4chem.github.io/nmr-predict"
    )
