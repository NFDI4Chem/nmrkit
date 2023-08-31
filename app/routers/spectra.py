from fastapi import APIRouter, HTTPException, status, UploadFile
from app.schemas import HealthCheck
import subprocess

router = APIRouter(
    prefix="/spectra",
    tags=["spectra"],
    dependencies=[],
    responses={404: {"description": "Not Found"}},
)


@router.get("/", include_in_schema=False)
@router.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check on Chem Module",
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


@router.post(
    "/parse",
    tags=["spectra"],
    summary="Parse the input spectra format and extract metadata",
    response_description="",
    status_code=status.HTTP_200_OK,
)
async def parse_spectra(file: UploadFile):
    """
    ## Parse the spectra file and extract meta-data
    Endpoint uses NMR-load-save to read the input spectra file (.jdx,.nmredata,.dx) and extracts metadata

    Returns:
        data: spectra data in JSON format
    """
    try:
        contents = file.file.read()
        file_path = "/tmp/" + file.filename
        with open(file_path, "wb") as f:
            f.write(contents)
        p = subprocess.Popen(
            "npx nmr-cli -p " + file_path, stdout=subprocess.PIPE, shell=True
        )
        (output, err) = p.communicate()
        p_status = p.wait()
        return output
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail="Error parsing the structure "
            + e.message
            + ". Error: "
            + err
            + ". Status:"
            + p_status,
            headers={"X-Error": "RDKit molecule input parse error"},
        )
    finally:
        file.file.close()
