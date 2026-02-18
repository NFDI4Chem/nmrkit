from fastapi import APIRouter, Body, HTTPException, status, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import io
from app.schemas import HealthCheck
from pydantic import BaseModel, HttpUrl, Field
from typing import Optional
import subprocess
import tempfile
import os
import json
from pathlib import Path

router = APIRouter(
    prefix="/spectra",
    tags=["spectra"],
    dependencies=[],
    responses={
        404: {"description": "Not Found"},
        408: {"description": "Processing timeout exceeded"},
        422: {"description": "Error parsing the spectra input"},
        500: {"description": "Docker or nmr-converter container not available"},
    },
)

# Container name for nmr-cli (from docker-compose.yml)
NMR_CLI_CONTAINER = "nmr-converter"


class UrlParseRequest(BaseModel):
    """Request model for parsing spectra from a remote URL."""

    url: HttpUrl = Field(
        ...,
        description="URL pointing to the NMR spectra file to parse",
        json_schema_extra={
            "examples": ["https://example.com/spectra/sample.jdx"],
        },
    )
    capture_snapshot: bool = Field(
        False,
        description="Generate an image snapshot of the spectra",
    )
    auto_processing: bool = Field(
        False,
        description="Enable automatic processing of spectrum (FID → FT spectra)",
    )
    auto_detection: bool = Field(
        False,
        description="Enable ranges and zones automatic detection",
    )
    raw_data: bool = Field(
        False, description="Include raw data in the output (default: data source)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "url": "https://example.com/spectra/sample.jdx",
                    "capture_snapshot": False,
                    "auto_processing": False,
                    "auto_detection": False,
                }
            ]
        }
    }


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


def run_command(
    file_path: str = None,
    url: str = None,
    capture_snapshot: bool = False,
    auto_processing: bool = False,
    auto_detection: bool = False,
    raw_data: bool = False,
) -> StreamingResponse:
    """Execute nmr-cli parse-spectra command in Docker container."""

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
    if raw_data:
        cmd.append("-r")

    try:
        result = subprocess.run(
            ["docker", "exec", NMR_CLI_CONTAINER] + cmd,
            capture_output=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=408, detail="Processing timeout exceeded")
    except FileNotFoundError:
        raise HTTPException(
            status_code=500, detail="Docker not found or nmr-converter container not running.")

    if result.returncode != 0:
        raise HTTPException(
            status_code=422,
            detail=f"NMR CLI error: {result.stderr.decode('utf-8') or 'Unknown error'}",
        )

    return StreamingResponse(
        io.BytesIO(result.stdout),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=parse-output.json"},
    )


def run_publication_string_command(publication_string: str) -> dict:
    """Execute nmr-cli parse-publication-string command in Docker container."""

    cmd = ["nmr-cli", "parse-publication-string", publication_string]

    try:
        result = subprocess.run(
            ["docker", "exec", NMR_CLI_CONTAINER] + cmd,
            capture_output=True,
            text=False,
            timeout=120
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=408,
            detail="Processing timeout exceeded"
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Docker not found or nmr-converter container not running."
        )

    if result.returncode != 0:
        error_msg = result.stderr.decode(
            "utf-8") if result.stderr else "Unknown error"
        raise HTTPException(
            status_code=422,
            detail=f"NMR CLI error: {error_msg}"
        )

    stdout = result.stdout.decode("utf-8").strip()

    if not stdout:
        raise HTTPException(
            status_code=422,
            detail="NMR CLI returned empty output. The publication string may be invalid or unrecognized."
        )

    # Validate that stdout is valid JSON without fully deserializing
    try:
        json.loads(stdout)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid JSON from NMR CLI: {e}"
        )

    return stdout


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
            status_code=500,
            detail=f"Failed to copy file to container: {error_msg}"
        )


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


class PeakItem(BaseModel):
    """A single NMR peak."""
    x: float = Field(..., description="Chemical shift in ppm")
    y: Optional[float] = Field(
        1.0, description="Peak intensity (default: 1.0)")
    width: Optional[float] = Field(
        1.0, description="Peak width in Hz (default: 1.0)")


class PeaksToNMRiumOptions(BaseModel):
    """Options for peaks-to-NMRium conversion."""
    nucleus: Optional[str] = Field(
        "1H", description="Nucleus type (e.g. '1H', '13C')")
    solvent: Optional[str] = Field("", description="NMR solvent")
    frequency: Optional[float] = Field(400, description="NMR frequency in MHz")
    nbPoints: Optional[int] = Field(
        131072, description="Number of points for spectrum generation", alias="nb_points")

    model_config = {"populate_by_name": True}


class PeaksToNMRiumRequest(BaseModel):
    """Request model for converting peaks to NMRium object."""
    peaks: list[PeakItem] = Field(
        ...,
        min_length=1,
        description="List of NMR peaks with chemical shift (x), intensity (y), and width",
    )
    options: Optional[PeaksToNMRiumOptions] = Field(
        None,
        description="Spectrum generation options",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "peaks": [
                        {"x": 7.26, "y": 1, "width": 1},
                        {"x": 2.10, "y": 1, "width": 1},
                    ],
                    "options": {
                        "nucleus": "1H",
                        "frequency": 400,
                    },
                }
            ]
        }
    }


def run_peaks_to_nmrium_command(payload: dict) -> str:
    """Execute nmr-cli peaks-to-nmrium command in Docker container via stdin."""

    cmd = ["docker", "exec", "-i", NMR_CLI_CONTAINER,
           "nmr-cli", "peaks-to-nmrium"]
    stdin_data = json.dumps(payload)

    try:
        result = subprocess.run(
            cmd,
            input=stdin_data.encode("utf-8"),
            capture_output=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=408,
            detail="Processing timeout exceeded",
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Docker not found or nmr-converter container not running.",
        )

    if result.returncode != 0:
        error_msg = result.stderr.decode(
            "utf-8") if result.stderr else "Unknown error"
        raise HTTPException(
            status_code=422,
            detail=f"NMR CLI error: {error_msg}",
        )

    stdout = result.stdout.decode("utf-8").strip()

    if not stdout:
        raise HTTPException(
            status_code=422,
            detail="NMR CLI returned empty output. The peak list may be invalid.",
        )

    try:
        json.loads(stdout)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid JSON from NMR CLI: {e}",
        )

    return stdout


@router.post(
    "/parse/file",
    tags=["spectra"],
    summary="Parse spectra from an uploaded file",
    description=(
        "Upload an NMR spectra file (JCAMP-DX, Bruker, etc.) and parse it into "
        "structured JSON. The file is processed by the **nmr-cli** tool running "
        "inside a Docker container.\n\n"
        "Supported file formats include JCAMP-DX (`.jdx`, `.dx`), Bruker directories "
        "(zipped), and other formats supported by nmr-cli."
    ),
    response_description="Parsed spectra data in NMRium-compatible JSON format",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Successfully parsed the spectra file"},
        408: {"description": "Processing timeout exceeded (120s limit)"},
        422: {"description": "Error parsing the spectra file"},
        500: {"description": "Docker or nmr-converter container not available"},
    },
)
async def parse_spectra_from_file(
    file: UploadFile = File(
        ..., description="NMR spectra file to parse (JCAMP-DX, Bruker zip, etc.)"),
    capture_snapshot: bool = Form(
        False,
        description="Generate an image snapshot of the spectra",
    ),
    auto_processing: bool = Form(
        False,
        description="Enable automatic processing of spectrum (FID → FT spectra)",
    ),
    auto_detection: bool = Form(
        False,
        description="Enable ranges and zones automatic detection",
    ),
    raw_data: bool = Form(
        False, description="Include raw data in the output (default: data source references)")
):
    """
    ## Parse spectra from an uploaded file

    Upload an NMR spectra file along with processing options using `multipart/form-data`.

    ### Processing Options
    | Option | Description |
    |--------|-------------|
    | `capture_snapshot` | Capture an image snapshot of the spectra |
    | `auto_processing` | Automatically process FID → FT spectra |
    | `auto_detection` | Automatically detect ranges and zones |
    | `raw_data` | Include raw data in the output (default: data source) |
    ### Returns
    Parsed spectra data in NMRium-compatible JSON format.
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
            raw_data=raw_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Error parsing the spectra file: {e}"
        )
    finally:
        if local_tmp_path and os.path.exists(local_tmp_path):
            os.unlink(local_tmp_path)
        if container_tmp_path:
            remove_file_from_container(container_tmp_path)
        await file.close()


@router.post(
    "/parse/url",
    tags=["spectra"],
    summary="Parse spectra from a remote URL",
    description=(
        "Provide a URL pointing to an NMR spectra file and parse it into structured "
        "JSON. The file is fetched and processed by the **nmr-cli** tool running "
        "inside a Docker container."
    ),
    response_description="Parsed spectra data in NMRium-compatible JSON format",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Successfully parsed the spectra from URL"},
        408: {"description": "Processing timeout exceeded (120s limit)"},
        422: {"description": "Error parsing spectra from the provided URL"},
        500: {"description": "Docker or nmr-converter container not available"},
    },
)
async def parse_spectra_from_url(request: UrlParseRequest):
    """
    ## Parse spectra from a remote URL

    Provide a URL to an NMR spectra file along with processing options in the JSON body.

    ### Processing Options
    | Option | Description |
    |--------|-------------|
    | `capture_snapshot` | Capture an image snapshot of the spectra |
    | `auto_processing` | Automatically process FID → FT spectra |
    | `auto_detection` | Automatically detect ranges and zones |
    | `raw_data` | Include raw data in the output (default: data source) |

    ### Returns
    Parsed spectra data in NMRium-compatible JSON format.
    """
    try:
        return run_command(
            url=str(request.url),
            capture_snapshot=request.capture_snapshot,
            auto_processing=request.auto_processing,
            auto_detection=request.auto_detection,
            raw_data=request.raw_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Error parsing spectra from URL: {e}"
        )


@router.post(
    "/parse/publication-string",
    tags=["spectra"],
    summary="Resurrect NMR spectrum from an ACS publication string",
    description=(
        "Parse an ACS-style NMR publication string and resurrect it into a full "
        "NMRium-compatible spectrum. The publication string is processed by the "
        "**nmr-cli** `parse-publication-string` command running inside a Docker "
        "container.\n\n"
        "The string is parsed to extract nucleus, solvent, and chemical shift ranges, "
        "which are then used to reconstruct the spectrum data (x/y arrays) at 400 MHz "
        "with 131072 points.\n\n"
        "### Example publication strings\n"
        "- `1H NMR (400 MHz, CDCl3) δ 7.26 (s, 1H), 2.10 (s, 3H)`\n"
        "- `13C NMR (101 MHz, DMSO-d6) δ 170.1, 136.5, 128.7`"
    ),
    response_description="Resurrected spectrum in NMRium-compatible JSON format",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Successfully resurrected spectrum from publication string"},
        408: {"description": "Processing timeout exceeded (120s limit)"},
        422: {"description": "Invalid publication string or NMR CLI error"},
        500: {"description": "Docker or nmr-converter container not available"},
    },
)
async def parse_publication_string(
    publication_string: str = Body(
        ...,
        media_type="text/plain",
        openapi_examples={
            "1H proton": {
                "summary": "1H NMR example",
                "value": "1H NMR (400 MHz, CDCl3) δ 7.26 (s, 1H), 2.10 (s, 3H)",
            },
            "13C carbon": {
                "summary": "13C NMR example",
                "value": "13C NMR (101 MHz, DMSO-d6) δ 170.1, 136.5, 128.7",
            },
        },
    ),
):
    """
    ## Resurrect NMR spectrum from a publication string

    Send the ACS-style NMR publication string directly as the request body
    (plain text, no JSON wrapping).

    ### Example request body
    ```
    1H NMR (400 MHz, CDCl3) δ 7.26 (s, 1H), 2.10 (s, 3H)
    ```

    ### Returns
    NMRium-compatible JSON with spectrum data, ranges, and metadata.
    """
    if not publication_string or not publication_string.strip():
        raise HTTPException(
            status_code=422,
            detail="Publication string cannot be empty."
        )

    try:
        raw_json = run_publication_string_command(publication_string.strip())
        return StreamingResponse(
            io.BytesIO(raw_json.encode("utf-8")),
            media_type="application/json",
            headers={
                "Content-Disposition": "attachment; filename=nmrium-spectrum.json",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Error parsing publication string: {e}"
        )


@router.post(
    "/parse/peaks",
    tags=["spectra"],
    summary="Convert a peak list to an NMRium-compatible spectrum",
    description=(
        "Convert a list of NMR peaks (chemical shifts with optional intensity and "
        "width) into a full NMRium-compatible spectrum object. Each peak is defined "
        "by its chemical shift position (`x` in ppm), intensity (`y`), and width "
        "(in Hz).\n\n"
        "The peaks are used to generate a simulated 1D spectrum using the "
        "**nmr-cli** `peaks-to-nmrium` command running inside a Docker container.\n\n"
        "### Example input\n"
        "```json\n"
        "{\n"
        '  "peaks": [\n'
        '    {"x": 7.26, "y": 1, "width": 1},\n'
        '    {"x": 2.10, "y": 1, "width": 1}\n'
        "  ],\n"
        '  "options": {\n'
        '    "nucleus": "1H",\n'
        '    "frequency": 400\n'
        "  }\n"
        "}\n"
        "```"
    ),
    response_description="Generated spectrum in NMRium-compatible JSON format",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Successfully generated spectrum from peak list"},
        408: {"description": "Processing timeout exceeded (120s limit)"},
        422: {"description": "Invalid peak list or NMR CLI error"},
        500: {"description": "Docker or nmr-converter container not available"},
    },
)
async def parse_peaks(request: PeaksToNMRiumRequest):
    """
    ## Convert a peak list to NMRium spectrum

    Provide a list of NMR peaks and optional generation parameters to produce
    an NMRium-compatible spectrum object.

    ### Peak fields
    | Field   | Type  | Required | Description                    |
    |---------|-------|----------|--------------------------------|
    | `x`     | float | Yes      | Chemical shift in ppm          |
    | `y`     | float | No       | Peak intensity (default: 1.0)  |
    | `width` | float | No       | Peak width in Hz (default: 1.0)|

    ### Options
    | Option      | Type   | Default | Description                |
    |-------------|--------|---------|----------------------------|
    | `nucleus`   | string | `1H`   | Nucleus type               |
    | `solvent`   | string | `""`   | NMR solvent                |
    | `frequency` | float  | `400`  | NMR frequency in MHz       |
    | `nb_points` | int    | `131072`| Number of spectrum points |

    ### Returns
    NMRium-compatible JSON with spectrum data and metadata.
    """
    if not request.peaks:
        raise HTTPException(
            status_code=422,
            detail="Peaks list cannot be empty.",
        )

    payload = {
        "peaks": [peak.model_dump() for peak in request.peaks],
    }
    if request.options:
        payload["options"] = {
            "nucleus": request.options.nucleus,
            "solvent": request.options.solvent,
            "frequency": request.options.frequency,
            "nbPoints": request.options.nbPoints,
        }

    try:
        raw_json = run_peaks_to_nmrium_command(payload)
        return StreamingResponse(
            io.BytesIO(raw_json.encode("utf-8")),
            media_type="application/json",
            headers={
                "Content-Disposition": "attachment; filename=nmrium-peaks.json",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Error converting peaks to NMRium: {e}",
        )
