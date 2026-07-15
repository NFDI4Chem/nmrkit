import subprocess
from fastapi import APIRouter, HTTPException, status, Response, Query
from app.schemas import HealthCheck

router = APIRouter(
    prefix="/convert",
    tags=["converter"],
    dependencies=[],
    responses={404: {"description": "Not Found"}},
)


@router.get("/", include_in_schema=False)
@router.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check on Converter Module",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    """
    ## Perform a Health Check
    Endpoint to perform a healthcheck on. This endpoint can primarily be used by Docker
    to ensure a robust container orchestration and management are in place. Other
    services that rely on the proper functioning of the API service will not deploy if this
    endpoint returns any other HTTP status code except 200 (OK).
    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    return HealthCheck(status="OK")


@router.get(
    "/spectra",
    tags=["converter"],
    summary="Convert NMR raw data to NMRium JSON",
    description=(
        "Fetch NMR raw data from a remote URL and convert it into "
        "[NMRium](https://www.nmrium.org/)-compatible JSON format. "
        "The conversion is performed by the **nmr-cli** tool running "
        "inside a Docker container.\n\n"
        "Supported input formats include Bruker, JCAMP-DX, and other "
        "formats recognized by nmr-cli."
    ),
    response_description="NMRium-compatible JSON representation of the NMR data",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Successfully converted NMR data to NMRium JSON",
            "content": {"application/json": {}},
        },
        500: {"description": "Conversion failed or Docker container not available"},
    },
)
async def nmr_load_save(
    url: str = Query(
        ...,
        description="URL pointing to the NMR raw data file to convert",
        examples=["https://example.com/nmr-data/sample.zip"],
    ),
):
    """
    ## Convert NMR raw data to NMRium JSON

    Fetches NMR raw data from the provided URL and converts it into NMRium JSON format.

    ### Parameters
    - **url**: A publicly accessible URL pointing to the NMR raw data

    ### Returns
    NMRium-compatible JSON object containing the converted spectra data.
    """
    process = subprocess.Popen(
        ["docker exec nmr-converter nmr-cli -u " + url],
        stdout=subprocess.PIPE,
        shell=True,
    )
    (output, err) = process.communicate()
    process.wait()
    if err:
        raise HTTPException(status_code=500, detail=err)
    else:
        return Response(content=output, media_type="application/json")
