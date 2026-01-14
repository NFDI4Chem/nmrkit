from fastapi import APIRouter, HTTPException, status, UploadFile, File, Query
from app.schemas import HealthCheck
import subprocess
import tempfile
import os
import json

router = APIRouter(
    prefix="/spectra",
    tags=["spectra"],
    dependencies=[],
    responses={404: {"description": "Not Found"}},
)

# Container name for nmr-cli (from docker-compose.yml)
NMR_CLI_CONTAINER = "nmr-converter"


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


CAPTURE_SNAPSHOT_QUERY = Query(
    False, alias="Capture snapshot", description="Generate a image snapshot of the spectra")
AUTO_PROCESSING_QUERY = Query(
    False, alias="Automatic processing",
    description="Enable automatic processing of spectrum (FID → FT spectra)"
)
AUTO_DETECTION_QUERY = Query(
    False, alias="Automatic detection",
    description="Enable ranges and zones automatic detection"
)


def run_command(
    file_path: Optional[str] = None,
    url: Optional[str] = None,
    capture_snapshot: bool = False,
    auto_processing: bool = False,
    auto_detection: bool = False,
) -> dict:

    cmd = ["nmr-cli", "parse-spectra"]

    if url:
        cmd.extend(["-u", url])
    elif file_path:
        cmd.extend(["-p", file_path])

    if capture_snapshot:
        cmd.append("-s")
    if auto_processing:
        cmd.append("-p")
    if auto_detection:
        cmd.append("-d")

    try:
        result = subprocess.run(
            ["docker", "exec", NMR_CLI_CONTAINER] + cmd,
            capture_output=True,
            text=False,
            timeout=120
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=408, detail="Processing timeout exceeded")
    except FileNotFoundError:
        raise HTTPException(
            status_code=500, detail="Docker not found or nmr-converter container not running.")

    if result.returncode != 0:
        error_msg = result.stderr.decode(
            "utf-8") if result.stderr else "Unknown error"
        raise HTTPException(
            status_code=422, detail=f"NMR CLI error: {error_msg}")

    # Parse output
    try:
        return json.loads(result.stdout.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500, detail=f"Invalid JSON from NMR CLI: {e}")


def copy_file_to_container(local_path: str, container_path: str) -> None:
    """Copy a file to the nmr-converter container."""
    try:
        subprocess.run(
            ["docker", "cp", local_path,
                f"{NMR_CLI_CONTAINER}:{container_path}"],
            check=True,
            capture_output=True,
            timeout=30
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode("utf-8") if e.stderr else "Unknown error"
        raise HTTPException(
            status_code=500, detail=f"Failed to copy file to container: {error_msg}")


def remove_file_from_container(container_path: str) -> None:
    """Remove a file from the nmr-converter container."""
    try:
        subprocess.run(
            ["docker", "exec", NMR_CLI_CONTAINER, "rm", "-f", container_path],
            capture_output=True,
            timeout=10
        )
    except Exception:
        pass


#  Parse from File Upload
@router.post(
    "/parse/file",
    tags=["spectra"],
    summary="Parse spectra from uploaded file",
    response_description="Spectra data in JSON format",
    status_code=status.HTTP_200_OK,
)
async def parse_spectra_from_file(
    file: UploadFile = File(...,
                            description="Upload a spectra file"),
    capture_snapshot: bool = CAPTURE_SNAPSHOT_QUERY,
    auto_processing: bool = AUTO_PROCESSING_QUERY,
    auto_detection: bool = AUTO_DETECTION_QUERY,
):
    """
    ## Parse spectra from uploaded file

    **Processing Options:**
    - `capture_snapshot (s)` : Capture snapshot of the spectra
    - `auto_processing  (p)` : Enable automatic processing of spectrum (FID → FT spectra)
    - `auto_detection   (d)` : Enable ranges and zones automatic detection

    Returns:
        Spectra data in JSON format
    """

    local_tmp_path = None
    container_tmp_path = None

    try:
        contents = await file.read()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(file.filename).suffix
        ) as tmp_file:
            tmp_file.write(contents)
            local_tmp_path = tmp_file.name

        container_tmp_path = f"/tmp/{Path(local_tmp_path).name}"

        # Copy file to nmr-converter container
        copy_file_to_container(local_tmp_path, container_tmp_path)

        # Run nmr-cli and get JSON output
        return run_command(
            file_path=container_tmp_path,
            capture_snapshot=capture_snapshot,
            auto_processing=auto_processing,
            auto_detection=auto_detection,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=422, detail=f"Error parsing the spectra file: {e}")
    finally:
        if local_tmp_path and os.path.exists(local_tmp_path):
            os.unlink(local_tmp_path)
        if container_tmp_path:
            remove_file_from_container(container_tmp_path)
        await file.close()


#  Parse from URL
@router.post(
    "/parse/url",
    tags=["spectra"],
    summary="Parse spectra from URL",
    response_description="Spectra data in JSON format",
    status_code=status.HTTP_200_OK,
)
async def parse_spectra_from_url(
    url: str = Query(..., alias="URL"),
    capture_snapshot: bool = CAPTURE_SNAPSHOT_QUERY,
    auto_processing: bool = AUTO_PROCESSING_QUERY,
    auto_detection: bool = AUTO_DETECTION_QUERY,
):
    """
    ## Parse spectra from URL

    **Processing Options:**
    - `capture_snapshot (s)` : Capture snapshot of the spectra
    - `auto_processing  (p)` : Enable automatic processing of spectrum (FID → FT spectra)
    - `auto_detection   (d)` : Enable ranges and zones automatic detection

    Returns:
        Spectra data in JSON format
    """
    if not url or not url.strip():
        raise HTTPException(
            status_code=400,
            detail="URL is required",
            headers={"X-Error": "No URL provided"},
        )

    try:
        output = run_command(
            url=url.strip(),
            capture_snapshot=capture_snapshot,
            auto_processing=auto_processing,
            auto_detection=auto_detection,
        )

        return output

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=422, detail=f"Error parsing spectra from URL: {e}")
