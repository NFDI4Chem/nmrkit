from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from app.schemas import HealthCheck
from pydantic import BaseModel, Field, model_validator
from typing import Annotated, List, Literal, Optional, Union
from enum import Enum
import subprocess
import json
import tempfile
import os
import uuid
import time
from pathlib import Path

router = APIRouter(
    prefix="/predict",
    tags=["predict"],
    dependencies=[],
    responses={404: {"description": "Not Found"}},
)

# Container name for nmr-cli (from docker-compose.yml)
NMR_CLI_CONTAINER = "nmr-converter"
SHARED_VOLUME_PATH = "/shared"


# ============================================================================
# ENUMS
# ============================================================================


class SpectraType(str, Enum):
    PROTON = "proton"
    CARBON = "carbon"
    COSY = "cosy"
    HSQC = "hsqc"
    HMBC = "hmbc"


class PeakShape(str, Enum):
    GAUSSIAN = "gaussian"
    LORENTZIAN = "lorentzian"


class Solvent(str, Enum):
    ANY = "Any"
    CHLOROFORM = "Chloroform-D1 (CDCl3)"
    DMSO = "Dimethylsulphoxide-D6 (DMSO-D6, C2D6SO)"
    METHANOL = "Methanol-D4 (CD3OD)"
    D2O = "Deuteriumoxide (D2O)"
    ACETONE = "Acetone-D6 ((CD3)2CO)"
    CCL4 = "TETRACHLORO-METHANE (CCl4)"
    PYRIDINE = "Pyridin-D5 (C5D5N)"
    BENZENE = "Benzene-D6 (C6D6)"
    NEAT = "neat"
    THF = "Tetrahydrofuran-D8 (THF-D8, C4D4O)"


# ============================================================================
# ENGINE-SPECIFIC OPTIONS
# ============================================================================


class FromTo(BaseModel):
    """Range with from/to values in ppm"""
    from_: float = Field(..., alias="from", description="From in ppm")
    to: float = Field(..., description="To in ppm")

    model_config = {"populate_by_name": True}


class NbPoints2D(BaseModel):
    x: int = Field(default=1024, description="2D spectrum X-axis points")
    y: int = Field(default=1024, description="2D spectrum Y-axis points")


class Options1D(BaseModel):
    """1D spectrum generation options for nmrdb.org"""
    proton: FromTo = Field(
        default=FromTo.model_validate({"from": -1, "to": 12}),
        description="Proton (1H) range in ppm",
    )
    carbon: FromTo = Field(
        default=FromTo.model_validate({"from": -5, "to": 220}),
        description="Carbon (13C) range in ppm",
    )
    nbPoints: int = Field(default=2**17, description="1D number of points")
    lineWidth: float = Field(default=1, description="1D line width")

    model_config = {"populate_by_name": True}


class Options2D(BaseModel):
    """2D spectrum generation options for nmrdb.org"""
    nbPoints: NbPoints2D = Field(
        default_factory=NbPoints2D,
        description="2D number of points",
    )


class NmrdbOptions(BaseModel):
    """Options for the nmrdb.org prediction engine"""
    name: str = Field(default="", description="Compound name")
    frequency: float = Field(default=400, description="NMR frequency (MHz)")
    one_d: Options1D = Field(
        alias="1d",
        default_factory=Options1D,
        description="1D spectrum options",
    )
    two_d: Options2D = Field(
        alias="2d",
        default_factory=Options2D,
        description="2D spectrum options",
    )
    autoExtendRange: bool = Field(
        default=True,
        description="Auto extend range if signals fall outside",
    )

    model_config = {"populate_by_name": True}


class NmrshiftOptions(BaseModel):
    """Options for the nmrshift prediction engine"""
    id: int = Field(default=1, description="Input ID")
    shifts: str = Field(default="1", description="Chemical shifts")
    solvent: Solvent = Field(default=Solvent.DMSO, description="NMR solvent")
    from_ppm: Optional[float] = Field(
        default=None,
        alias="from",
        description="From in (ppm) for spectrum generation",
    )
    to_ppm: Optional[float] = Field(
        default=None,
        alias="to",
        description="To in (ppm) for spectrum generation",
    )
    nbPoints: int = Field(default=1024, description="Number of points")
    lineWidth: float = Field(default=1, description="Line width")
    frequency: float = Field(default=400, description="NMR frequency (MHz)")
    tolerance: float = Field(
        default=0.001, description="Tolerance to group peaks")
    peakShape: PeakShape = Field(
        default=PeakShape.LORENTZIAN, description="Peak shape")

    model_config = {"populate_by_name": True}


# ============================================================================
# REQUEST MODELS
# ============================================================================


NMRDB_SUPPORTED_SPECTRA = {"proton", "carbon", "cosy", "hsqc", "hmbc"}
NMRSHIFT_SUPPORTED_SPECTRA = {"proton", "carbon"}


class NmrdbPredictRequest(BaseModel):
    """Prediction request using the nmrdb.org engine"""
    engine: Literal["nmrdb.org"] = Field(..., description="Prediction engine")
    structure: str = Field(..., description="MOL file content")
    spectra: List[SpectraType] = Field(...,
                                       description="Spectra types", min_length=1)
    options: NmrdbOptions = Field(default_factory=NmrdbOptions)

    @model_validator(mode="after")
    def validate_spectra(self):
        unsupported = [
            s.value for s in self.spectra if s.value not in NMRDB_SUPPORTED_SPECTRA]
        if unsupported:
            raise ValueError(
                f"nmrdb.org does not support: {unsupported}. "
                f"Supported: {sorted(NMRDB_SUPPORTED_SPECTRA)}"
            )
        return self


class NmrshiftPredictRequest(BaseModel):
    """Prediction request using the nmrshift engine"""
    engine: Literal["nmrshift"] = Field(..., description="Prediction engine")
    structure: str = Field(..., description="MOL file content")
    spectra: List[SpectraType] = Field(...,
                                       description="Spectra types", min_length=1)
    options: NmrshiftOptions = Field(default_factory=NmrshiftOptions)

    @model_validator(mode="after")
    def validate_spectra(self):
        unsupported = [
            s.value for s in self.spectra if s.value not in NMRSHIFT_SUPPORTED_SPECTRA]
        if unsupported:
            raise ValueError(
                f"nmrshift does not support: {unsupported}. "
                f"Supported: {sorted(NMRSHIFT_SUPPORTED_SPECTRA)}"
            )
        return self


# File upload request models - same options as structure models
class NmrdbFileRequest(BaseModel):
    """File upload prediction request using the nmrdb.org engine"""
    engine: Literal["nmrdb.org"] = Field(..., description="Prediction engine")
    spectra: List[SpectraType] = Field(...,
                                       description="Spectra types", min_length=1)
    options: NmrdbOptions = Field(default_factory=NmrdbOptions)

    @model_validator(mode="after")
    def validate_spectra(self):
        unsupported = [
            s.value for s in self.spectra if s.value not in NMRDB_SUPPORTED_SPECTRA]
        if unsupported:
            raise ValueError(
                f"nmrdb.org does not support: {unsupported}. "
                f"Supported: {sorted(NMRDB_SUPPORTED_SPECTRA)}"
            )
        return self


class NmrshiftFileRequest(BaseModel):
    """File upload prediction request using the nmrshift engine"""
    engine: Literal["nmrshift"] = Field(..., description="Prediction engine")
    spectra: List[SpectraType] = Field(...,
                                       description="Spectra types", min_length=1)
    options: NmrshiftOptions = Field(default_factory=NmrshiftOptions)

    @model_validator(mode="after")
    def validate_spectra(self):
        unsupported = [
            s.value for s in self.spectra if s.value not in NMRSHIFT_SUPPORTED_SPECTRA]
        if unsupported:
            raise ValueError(
                f"nmrshift does not support: {unsupported}. "
                f"Supported: {sorted(NMRSHIFT_SUPPORTED_SPECTRA)}"
            )
        return self


PredictRequest = Annotated[
    Union[NmrdbPredictRequest, NmrshiftPredictRequest],
    Field(discriminator="engine"),
]

FileRequest = Annotated[
    Union[NmrdbFileRequest, NmrshiftFileRequest],
    Field(discriminator="engine"),
]


# ============================================================================
# CLI BUILDERS
# ============================================================================


def build_nmrdb_args(options: NmrdbOptions, spectra: List[SpectraType]) -> list[str]:
    """Build CLI arguments for nmrdb.org"""
    one_d = options.one_d
    two_d = options.two_d

    args = [
        "--engine", "nmrdb.org",
        "--spectra", *[s.value for s in spectra],
        "--frequency", str(options.frequency),
        "--protonFrom", str(one_d.proton.from_),
        "--protonTo", str(one_d.proton.to),
        "--carbonFrom", str(one_d.carbon.from_),
        "--carbonTo", str(one_d.carbon.to),
        "--nbPoints1d", str(one_d.nbPoints),
        "--lineWidth", str(one_d.lineWidth),
        "--nbPoints2dX", str(two_d.nbPoints.x),
        "--nbPoints2dY", str(two_d.nbPoints.y),
    ]

    if options.name:
        args.extend(["--name", options.name])
    if not options.autoExtendRange:
        args.append("--no-autoExtendRange")

    return args


def build_nmrshift_args(options: NmrshiftOptions, spectra: List[SpectraType]) -> list[str]:
    """Build CLI arguments for nmrshift"""
    args = [
        "--engine", "nmrshift",
        "--spectra", *[s.value for s in spectra],
        "--id", str(options.id),
        "--shifts", options.shifts,
        "--solvent", options.solvent.value,
        "--nbPoints", str(options.nbPoints),
        "--lineWidth", str(options.lineWidth),
        "--frequency", str(options.frequency),
        "--tolerance", str(options.tolerance),
        "--peakShape", options.peakShape.value,
    ]

    if options.from_ppm is not None:
        args.extend(["--from", str(options.from_ppm)])
    if options.to_ppm is not None:
        args.extend(["--to", str(options.to_ppm)])

    return args


def build_cli_args(request: Union[NmrdbPredictRequest, NmrshiftPredictRequest,
                                  NmrdbFileRequest, NmrshiftFileRequest]) -> list[str]:
    """Build CLI args from any request type"""
    if isinstance(request, (NmrdbPredictRequest, NmrdbFileRequest)):
        return build_nmrdb_args(request.options, request.spectra)
    elif isinstance(request, (NmrshiftPredictRequest, NmrshiftFileRequest)):
        return build_nmrshift_args(request.options, request.spectra)
    else:
        raise HTTPException(
            status_code=400, detail=f"Unknown engine type: {type(request)}")


# ============================================================================
# HELPERS
# ============================================================================


def copy_file_to_container(local_path: str, container_path: str) -> None:
    """Copy file to container"""
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
            status_code=500, detail=f"Failed to copy file: {error_msg}")


def remove_file_from_container(container_path: str) -> None:
    """Remove file from container"""
    try:
        subprocess.run(
            ["docker", "exec", NMR_CLI_CONTAINER, "rm", "-f", container_path],
            capture_output=True,
            timeout=10
        )
    except Exception:
        pass


def execute_cli(cmd: list[str], engine: str) -> dict:
    """Execute CLI command and return parsed JSON"""
    timeout = 300 if engine == "nmrdb.org" else 120
    start_time = time.time()

    try:
        result = subprocess.run(
            ["docker", "exec", NMR_CLI_CONTAINER] + cmd,
            capture_output=True,
            text=False,
            timeout=timeout
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=408,
            detail={
                "message": f"Prediction timed out after {timeout}s",
                "engine": engine,
                "hint": "nmrdb.org predictions can take 30-60s, try again or use nmrshift for faster results",
            }
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Docker not found or nmr-converter container is not running",
                "hint": "Run: docker compose up -d",
            }
        )

    elapsed = round(time.time() - start_time, 2)
    stdout = result.stdout.decode("utf-8", errors="replace").strip()
    stderr = result.stderr.decode("utf-8", errors="replace").strip()

    if result.returncode != 0:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "NMR CLI command failed",
                "engine": engine,
                "exit_code": result.returncode,
                "error": stderr or "No error output from CLI",
                "elapsed_seconds": elapsed,
            }
        )

    if not stdout:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "NMR CLI returned empty output",
                "engine": engine,
                "exit_code": result.returncode,
                "stderr": stderr or "No error output from CLI",
                "elapsed_seconds": elapsed,
                "hint": "Check that all required CLI arguments are valid",
            }
        )

    # Strip any warning/info messages printed before the JSON output
    json_start = stdout.find('{')
    if json_start > 0:
        warnings = stdout[:json_start].strip()
        print(f"[WARN] CLI warnings before JSON: {warnings}")
        stdout = stdout[json_start:]

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "NMR CLI returned invalid JSON",
                "engine": engine,
                "parse_error": str(e),
                "stdout_preview": stdout[:500],
                "stderr": stderr or "No error output from CLI",
                "elapsed_seconds": elapsed,
            }
        )


def run_predict_command(structure: str, cli_args: list[str], engine: str) -> dict:
    """Execute nmr-cli predict with structure string"""
    # CRITICAL: Escape newlines for CLI
    structure_escaped = structure.replace('\n', '\\n')
    cmd = ["nmr-cli", "predict", "-s", structure_escaped] + cli_args
    return execute_cli(cmd, engine)


def run_predict_command_with_file(file_path: str, cli_args: list[str], engine: str) -> dict:
    """Execute nmr-cli predict with file"""
    cmd = ["nmr-cli", "predict", "--file", file_path] + cli_args
    return execute_cli(cmd, engine)


# ============================================================================
# HEALTH CHECK
# ============================================================================


@router.get("/", include_in_schema=False)
@router.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check on Predict Module",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    """Health check endpoint"""
    return HealthCheck(status="OK")


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post(
    "/",
    tags=["predict"],
    summary="Predict NMR spectra from MOL string",
    response_description="Predicted spectra in NMRium JSON format",
    status_code=status.HTTP_200_OK,
)
async def predict_from_structure(request: PredictRequest):
    """
    ## Predict NMR spectra from MOL string

    **Note:** nmrdb.org predictions take 30-60s. Use curl/Postman, not Swagger.

    **Engines:**
    - **nmrshift** — Supports: proton, carbon
    - **nmrdb.org** — Supports: proton, carbon, cosy, hsqc, hmbc

    **Example (nmrshift):**
    ```json
    {
        "engine": "nmrshift",
        "structure": "\\n  Mrv2311...\\nM  END",
        "spectra": ["proton"],
        "options": {
            "solvent": "Chloroform-D1 (CDCl3)",
            "frequency": 400,
            "nbPoints": 1024,
            "lineWidth": 1,
            "peakShape": "lorentzian"
        }
    }
    ```

    **Example (nmrdb.org):**
    ```json
    {
        "engine": "nmrdb.org",
        "structure": "\\n  Mrv2311...\\nM  END",
        "spectra": ["proton", "carbon"],
        "options": {
            "name": "Benzene",
            "frequency": 400,
            "1d": {
                "proton": {"from": -1, "to": 12},
                "carbon": {"from": -5, "to": 220},
                "nbPoints": 131072,
                "lineWidth": 1
            },
            "autoExtendRange": true
        }
    }
    ```
    """
    try:
        cli_args = build_cli_args(request)
        return run_predict_command(request.structure, cli_args, request.engine)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error: {e}")


@router.post(
    "/file",
    tags=["predict"],
    summary="Predict NMR spectra from uploaded MOL file",
    response_description="Predicted spectra in NMRium JSON format",
    status_code=status.HTTP_200_OK,
)
async def predict_from_file(
    file: UploadFile = File(..., description="MOL file"),
    request: str = Form(..., description="""JSON string with engine, spectra and options. Examples:

nmrshift: {"engine": "nmrshift", "spectra": ["proton"], "options": {"solvent": "Chloroform-D1 (CDCl3)", "frequency": 400, "nbPoints": 1024, "lineWidth": 1, "peakShape": "lorentzian"}}

nmrdb.org: {"engine": "nmrdb.org", "spectra": ["proton", "carbon"], "options": {"name": "Benzene", "frequency": 400, "1d": {"proton": {"from": -1, "to": 12}, "carbon": {"from": -5, "to": 220}, "nbPoints": 131072, "lineWidth": 1}, "autoExtendRange": true}}
"""),
):
    """
    ## Predict NMR spectra from uploaded MOL file

    Upload a MOL file and pass engine options as a JSON string in the `request` field.

    **Note:** nmrdb.org predictions take 30-60s. Use curl/Postman, not Swagger.

    **nmrshift example request field:**
    ```json
    {
        "engine": "nmrshift",
        "spectra": ["proton"],
        "options": {
            "solvent": "Chloroform-D1 (CDCl3)",
            "frequency": 400,
            "nbPoints": 1024,
            "lineWidth": 1,
            "peakShape": "lorentzian"
        }
    }
    ```

    **nmrdb.org example request field:**
    ```json
    {
        "engine": "nmrdb.org",
        "spectra": ["proton", "carbon"],
        "options": {
            "name": "Benzene",
            "frequency": 400,
            "1d": {
                "proton": {"from": -1, "to": 12},
                "carbon": {"from": -5, "to": 220},
                "nbPoints": 131072,
                "lineWidth": 1
            },
            "autoExtendRange": true
        }
    }
    ```
    """
    local_file_path = None
    container_file_path = None
    use_shared_volume = os.path.exists(
        SHARED_VOLUME_PATH) and os.access(SHARED_VOLUME_PATH, os.W_OK)

    try:
        # Parse the JSON request field
        try:
            request_data = json.loads(request)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=422, detail=f"Invalid JSON in request field: {e}")

        # Validate against the correct model based on engine
        engine = request_data.get("engine")
        if engine == "nmrdb.org":
            parsed_request = NmrdbFileRequest(**request_data)
        elif engine == "nmrshift":
            parsed_request = NmrshiftFileRequest(**request_data)
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown engine: {engine}. Use 'nmrdb.org' or 'nmrshift'")

        # Build CLI args using same builders as structure endpoint
        cli_args = build_cli_args(parsed_request)

        # Read and save uploaded file
        contents = await file.read()

        if use_shared_volume:
            # FAST: Write directly to shared volume
            filename = f"predict_{uuid.uuid4().hex[:8]}.mol"
            local_file_path = os.path.join(SHARED_VOLUME_PATH, filename)
            container_file_path = f"/shared/{filename}"
            with open(local_file_path, 'wb') as f:
                f.write(contents)
        else:
            # Fallback: Use temp file + docker cp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mol") as tmp_file:
                tmp_file.write(contents)
                local_file_path = tmp_file.name
            container_file_path = f"/tmp/{Path(local_file_path).name}"
            copy_file_to_container(local_file_path, container_file_path)

        return run_predict_command_with_file(container_file_path, cli_args, parsed_request.engine)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error: {e}")
    finally:
        if local_file_path and os.path.exists(local_file_path):
            try:
                os.unlink(local_file_path)
            except Exception:
                pass
        if not use_shared_volume and container_file_path:
            remove_file_from_container(container_file_path)
        await file.close()
