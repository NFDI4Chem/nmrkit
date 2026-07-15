from typing import List, Annotated, Union
from app.core.config import LWREG_CONFIG
from fastapi import APIRouter, HTTPException, Body, Query, status
from lwreg.utils import (
    initdb,
    bulk_register,
    query,
    retrieve,
    RegistrationFailureReasons,
)
from rdkit import Chem
from io import BytesIO
from app.schemas import HealthCheck

router = APIRouter(
    prefix="/registration",
    tags=["registration"],
    dependencies=[],
    responses={
        404: {"description": "Not Found"},
        500: {"description": "Internal server error"},
    },
)


@router.get("/", include_in_schema=False)
@router.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check on Registration Module",
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
    services which rely on the proper functioning of the API service will not deploy if this
    endpoint returns any other HTTP status code except 200 (OK).
    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    return HealthCheck(status="OK")


@router.post(
    "/init",
    tags=["registration"],
    summary="Initialize the registration database",
    description=(
        "Initialize (or re-initialize) the molecule registration database. "
        "**Warning:** This operation destroys all existing data in the registration "
        "database. Set `confirm` to `true` to proceed."
    ),
    response_description="Boolean indicating whether initialization was successful",
    status_code=status.HTTP_200_OK,
    response_model=Union[bool, None],
    responses={
        200: {
            "description": "Database initialized successfully",
            "content": {"application/json": {"example": True}},
        },
    },
)
async def initialise_database(
    confirm: Annotated[
        bool,
        Body(
            embed=True,
            description="Set to true to confirm database initialization. False returns immediately.",
        ),
    ] = False,
):
    """
    ## Initialize the registration database

    > **WARNING:** This call destroys any existing information in the registration database.

    ### Parameters
    - **confirm**: Must be set to `true` to actually perform the initialization.
      If `false` (default), the call returns immediately without changes.

    ### Returns
    `true` if initialization was successful, `null` if confirm was `false`.
    """
    return initdb(config=LWREG_CONFIG, confirm=confirm)


@router.post(
    "/register",
    tags=["registration"],
    summary="Register new molecules",
    description=(
        "Register one or more molecules in the database. Accepts SMILES strings "
        "(one per line) or an SDF block as plain text. Returns the new registry "
        "numbers (molregnos) for successfully registered molecules.\n\n"
        "Duplicate molecules are flagged as `DUPLICATE` and parse failures as "
        "`PARSE_FAILURE` in the response array."
    ),
    response_description="Array of registry numbers (integers) or status strings (DUPLICATE, PARSE_FAILURE)",
    status_code=status.HTTP_200_OK,
    response_model=List[Union[str, int]],
    responses={
        200: {
            "description": "Molecules registered successfully",
            "content": {
                "application/json": {
                    "example": [1, "DUPLICATE", 3],
                }
            },
        },
        422: {"description": "Registration failed — all entries are duplicates or unparseable"},
    },
)
async def register_compounds(
    data: Annotated[
        str,
        Body(
            embed=False,
            media_type="text/plain",
            description=(
                "Molecular data as plain text. Provide either SMILES strings "
                "(one per line) or an SDF block (containing $$$$ delimiters)."
            ),
            openapi_examples={
                "smiles": {
                    "summary": "SMILES input",
                    "description": "One or more SMILES strings, one per line",
                    "value": "CCCC\nCCCCO\nc1ccccc1",
                },
                "sdf": {
                    "summary": "SDF block",
                    "description": "An SDF block with $$$$ delimiters",
                    "value": (
                        "\n  Mrv2311 08092305412D\n\n"
                        "  3  2  0  0  0  0            999 V2000\n"
                        "   -0.4018    0.6926    0.0000 C   0  0  0  0  0  0\n"
                        "    0.3127    1.1051    0.0000 C   0  0  0  0  0  0\n"
                        "    1.0272    0.6926    0.0000 O   0  0  0  0  0  0\n"
                        "  1  2  1  0  0  0  0\n"
                        "  2  3  1  0  0  0  0\n"
                        "M  END\n$$$$"
                    ),
                },
            },
        ),
    ] = "CCCC",
):
    """
    ## Register new molecules

    Registers one or more molecules (assuming they don't already exist) and returns
    the new registry number(s) (molregno).

    ### Input Formats
    - **SMILES** — one SMILES string per line
    - **SDF block** — MOL/SDF format with `$$$$` delimiters

    ### Response
    An array where each element is either:
    - An **integer** — the new molregno for a successfully registered molecule
    - `"DUPLICATE"` — the molecule already exists in the database
    - `"PARSE_FAILURE"` — the molecule could not be parsed
    """
    try:
        if "$$$$" in data:
            molStream = BytesIO(data.encode("utf8"))
            mols = [m for m in Chem.ForwardSDMolSupplier(molStream)]
        else:
            smiles = data.splitlines()
            mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        if len(mols) > 0:
            db_responses = bulk_register(mols=mols, config=LWREG_CONFIG)
            reg_responses = []
            ops_total_failure = True
            for res in db_responses:
                if res == RegistrationFailureReasons.PARSE_FAILURE:
                    reg_responses.append("PARSE_FAILURE")
                elif res == RegistrationFailureReasons.DUPLICATE:
                    reg_responses.append("DUPLICATE")
                else:
                    ops_total_failure = False
                    reg_responses.append(res)
            if ops_total_failure:
                raise
            else:
                return reg_responses
        else:
            raise
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail="Registration failed: ALL_DUPLICATE_ENTRIES " + e.message,
        )


@router.get(
    "/query",
    tags=["registration"],
    summary="Query if a molecule is already registered",
    description=(
        "Check whether a molecule (given as a SMILES string) has already been "
        "registered in the database. Returns the corresponding registry numbers "
        "(molregnos) if found."
    ),
    response_model=List[int],
    response_description="Array of matching registry numbers (molregnos)",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Query completed successfully",
            "content": {
                "application/json": {
                    "example": [42, 108],
                }
            },
        },
        500: {"description": "Internal server error during query"},
    },
)
async def query_compounds(
    smi: str = Query(
        ...,
        description="SMILES string of the molecule to query",
        examples=["CCCC", "c1ccccc1", "CCO"],
    ),
):
    """
    ## Query if a molecule is already registered

    Checks the registration database for the given molecule.

    ### Parameters
    - **smi**: A valid SMILES string

    ### Returns
    An array of integer registry numbers (molregnos) matching the query.
    Returns an empty array if the molecule is not registered.
    """
    try:
        res = query(smiles=smi, config=LWREG_CONFIG)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error" + e.message)


@router.post(
    "/retrieve",
    tags=["registration"],
    summary="Retrieve registered molecules by ID",
    description=(
        "Retrieve one or more registered molecules by their registry IDs (molregnos). "
        "Returns the molecular data and format for each requested ID."
    ),
    response_model=tuple(),
    response_description="Array of (molregno, data, format) tuples for each requested ID",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Successfully retrieved molecule data",
            "content": {
                "application/json": {
                    "example": [
                        [1, "CCCC", "smiles"],
                        [2, "CCO", "smiles"],
                    ],
                }
            },
        },
        500: {"description": "Internal server error during retrieval"},
    },
)
async def retrieve_compounds(
    ids: List[int] = Body(
        ...,
        description="List of registry IDs (molregnos) to retrieve",
        examples=[[1, 2, 3]],
    ),
):
    """
    ## Retrieve registered molecules by ID

    Fetches molecule data for one or more registry IDs (molregnos).

    ### Request Body
    A JSON array of integer registry IDs.

    ### Returns
    An array of `[molregno, data, format]` tuples containing:
    - **molregno** — the registry number
    - **data** — the molecular data (SMILES, MOL block, etc.)
    - **format** — the format of the data
    """
    try:
        res = retrieve(ids=ids, config=LWREG_CONFIG)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error" + e.message)
