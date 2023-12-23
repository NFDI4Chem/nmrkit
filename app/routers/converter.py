import subprocess
from fastapi import APIRouter, HTTPException, status, Response
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
    summary="Load and convert NMR raw data",
    # response_model=List[int],
    response_description="Load and convert NMR raw data",
    status_code=status.HTTP_200_OK,
)
async def nmr_load_save(url: str):
    """
    ## Return nmrium json

    Returns:
        Return nmrium json
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
