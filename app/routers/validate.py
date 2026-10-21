from fastapi import APIRouter, HTTPException, status
from app.schemas import HealthCheck
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional
from collections import OrderedDict
import hashlib
import json
import subprocess
import threading
import time

router = APIRouter(
    prefix="/validate",
    tags=["validate"],
    dependencies=[],
    responses={
        404: {"description": "Not Found"},
        408: {"description": "Validation timed out"},
        422: {"description": "Invalid assignment set or structure"},
        500: {"description": "Docker or nmr-converter container not available"},
        503: {"description": "nmrshiftdb2 quickcheck is unavailable"},
    },
)

# Container name for nmr-cli (from docker-compose.yml)
NMR_CLI_CONTAINER = "nmr-converter"
VALIDATION_TIMEOUT_SECONDS = 120

# nmr-cli validate-assignments exit codes
EXIT_INVALID_INPUT = 2
EXIT_QUICKCHECK_UNAVAILABLE = 3

CACHE_TTL_SECONDS = 24 * 60 * 60
CACHE_MAX_ENTRIES = 512


# ============================================================================
# REQUEST MODELS
# ============================================================================


Nucleus = Literal["13C", "1H"]


class Structure(BaseModel):
    molfile: str = Field(..., min_length=1,
                         description="V2000 MOL block; atom numbers in assignments refer to it")
    source: Optional[str] = Field(
        default=None, description="Origin of the structure, e.g. nmrium, mnova, nmredata, manual")


class Conditions(BaseModel):
    solvent: Optional[str] = Field(
        default=None, description="Solvent name or abbreviation, e.g. CDCl3 or DMSO-d6")
    temperature_k: Optional[float] = Field(default=None, description="Temperature in K")
    frequency_mhz: Optional[Dict[Nucleus, float]] = Field(
        default=None, description="Spectrometer frequency per nucleus")


class Assignment(BaseModel):
    nucleus: Nucleus
    atoms: List[int] = Field(
        ...,
        description=(
            "1-based molfile atom indices. For 1H, an index may be the carrying heavy atom "
            "or an explicit H atom. An empty list marks an observed but unassigned signal."
        ),
    )
    label: Optional[str] = Field(default=None, description="Author label, e.g. C-2, C-6 or H-5'a")
    shift: float = Field(..., description="Observed chemical shift in ppm")
    multiplicity: Optional[str] = None
    n_h: Optional[int] = Field(default=None, ge=0, description="Integral in protons")
    diastereotopic: Optional[Literal["a", "b"]] = None


class UnassignedPeak(BaseModel):
    nucleus: Nucleus
    shift: float
    kind: Literal["solvent", "impurity", "unknown"] = "unknown"


class Tolerance(BaseModel):
    ok: float = Field(..., gt=0)
    fail: float = Field(..., gt=0)


class ValidationOptions(BaseModel):
    fallback_tolerances: Optional[Dict[Nucleus, Tolerance]] = None


class AssignmentValidationRequest(BaseModel):
    structure: Structure
    conditions: Optional[Conditions] = None
    assignments: List[Assignment] = Field(..., min_length=1)
    unassigned_peaks: Optional[List[UnassignedPeak]] = None
    options: Optional[ValidationOptions] = None


# ============================================================================
# CACHE
# ============================================================================


class ReportCache:
    """Small in-process TTL cache; the quickcheck servlet is a shared public service."""

    def __init__(self, ttl: int, max_entries: int):
        self.ttl = ttl
        self.max_entries = max_entries
        self._entries: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, report = entry
            if time.time() - stored_at > self.ttl:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return report

    def put(self, key: str, report: dict) -> None:
        with self._lock:
            self._entries[key] = (time.time(), report)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


report_cache = ReportCache(CACHE_TTL_SECONDS, CACHE_MAX_ENTRIES)


def request_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ============================================================================
# CLI
# ============================================================================


def parse_cli_error(stderr: str) -> Dict[str, Any]:
    start = stderr.find("{")
    if start >= 0:
        try:
            return json.loads(stderr[start:])
        except json.JSONDecodeError:
            pass
    return {"message": stderr or "No error output from CLI"}


def run_validate_command(payload: dict) -> dict:
    """Pipe the assignment set to `nmr-cli validate-assignments` and return the report."""
    try:
        result = subprocess.run(
            ["docker", "exec", "-i", NMR_CLI_CONTAINER, "nmr-cli", "validate-assignments"],
            input=json.dumps(payload).encode("utf-8"),
            capture_output=True,
            timeout=VALIDATION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=408,
            detail={"message": f"Validation timed out after {VALIDATION_TIMEOUT_SECONDS}s"},
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Docker not found or nmr-converter container is not running",
                "hint": "Run: docker compose up -d",
            },
        )

    stdout = result.stdout.decode("utf-8", errors="replace").strip()
    stderr = result.stderr.decode("utf-8", errors="replace").strip()

    if result.returncode == EXIT_INVALID_INPUT:
        raise HTTPException(status_code=422, detail=parse_cli_error(stderr))
    if result.returncode == EXIT_QUICKCHECK_UNAVAILABLE:
        raise HTTPException(status_code=503, detail=parse_cli_error(stderr))
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "NMR CLI command failed",
                "exit_code": result.returncode,
                "error": parse_cli_error(stderr),
            },
        )

    json_start = stdout.find("{")
    try:
        return json.loads(stdout[json_start:] if json_start >= 0 else stdout)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "NMR CLI returned invalid JSON",
                "parse_error": str(e),
                "stdout_preview": stdout[:500],
            },
        )


# ============================================================================
# HEALTH CHECK
# ============================================================================


@router.get("/", include_in_schema=False)
@router.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check on Validate Module",
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
    "/assignments",
    summary="Validate 1H/13C assignments against nmrshiftdb2 quickcheck",
    description=(
        "Submit a structure with assigned 1H and 13C shifts. Each nucleus is checked "
        "against nmrshiftdb2 HOSE-code predictions in a single quickcheck request.\n\n"
        "The response has two layers:\n\n"
        "| Layer | Question it answers |\n"
        "|-------|--------------------|\n"
        "| `reports` | How well do the assigned shifts fit, atom by atom? (quality report on the author's assignments: mark 1–10, per-atom deviation, HOSE spheres) |\n"
        "| `assignment_check` | Are the shifts on the right atoms? (per-assignment status, swap suggestions, equivalence, missing signals, referencing offset) |\n\n"
        "`verdict` combines both, 13C first. Predictions with fewer than 4 HOSE spheres can "
        "at most lead to `review`, never `fail`. The mark formula approximates nmrshiftdb2 "
        "and is flagged with `mark_is_approximate`."
    ),
    response_description="Validation report",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Validation report"},
        408: {"description": "Validation timed out"},
        422: {"description": "Invalid assignment set or structure"},
        503: {"description": "nmrshiftdb2 quickcheck is unavailable, retry later"},
    },
)
def validate_assignments(request: AssignmentValidationRequest) -> Dict[str, Any]:
    """
    ## Validate NMR assignments

    ### Example
    ```json
    {
        "structure": {"molfile": "\\n  Mnova...\\nM  END", "source": "mnova"},
        "conditions": {"solvent": "CDCl3"},
        "assignments": [
            {"nucleus": "13C", "atoms": [1, 3], "label": "C-2, C-6", "shift": 106.65},
            {"nucleus": "13C", "atoms": [5], "label": "C-4", "shift": 143.52},
            {"nucleus": "1H", "atoms": [1, 3], "label": "H-2, H-6", "shift": 7.12, "n_h": 2}
        ],
        "unassigned_peaks": [{"nucleus": "13C", "shift": 77.16, "kind": "solvent"}]
    }
    ```
    """
    payload = request.model_dump(exclude_none=True)
    key = request_hash(payload)

    cached = report_cache.get(key)
    if cached is not None:
        return {**cached, "cached": True}

    report = run_validate_command(payload)
    report_cache.put(key, report)
    return {**report, "cached": False}
