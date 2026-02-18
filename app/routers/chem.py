from typing import Annotated
from psycopg2.errors import UniqueViolation
from app.modules.cdkmodules import getCDKHOSECodes
from fastapi import APIRouter, HTTPException, status, Query
from app.modules.rdkitmodules import getRDKitHOSECodes
from app.schemas import HealthCheck

router = APIRouter(
    prefix="/chem",
    tags=["chem"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
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
    Endpoint to perform a healthcheck on. This endpoint can primarily be used Docker
    to ensure a robust container orchestration and management is in place. Other
    services which rely on proper functioning of the API service will not deploy if this
    endpoint returns any other HTTP status code except 200 (OK).
    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    return HealthCheck(status="OK")


@router.get(
    "/hosecode",
    tags=["chem"],
    summary="Generate HOSE codes for a molecule",
    description=(
        "Generate **Hierarchically Ordered Spherical Environment (HOSE)** codes "
        "for every atom in the given molecule. HOSE codes encode the local chemical "
        "environment around each atom up to a configurable number of spheres.\n\n"
        "Supports two cheminformatics frameworks:\n"
        "- **CDK** (Chemistry Development Kit) — default, supports stereo\n"
        "- **RDKit** — alternative implementation"
    ),
    response_model=list[str],
    response_description="Array of HOSE code strings, one per atom in the molecule",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Successfully generated HOSE codes",
            "content": {
                "application/json": {
                    "example": [
                        "C(CC,CC,&)",
                        "C(CC,C&,&)",
                        "C(CC,CC,&)",
                        "C(CCC,CC&,&)",
                        "C(CC,CC,CC)",
                        "C(CC,CC,CC)",
                    ]
                }
            },
        },
        409: {"description": "Molecule already exists (unique constraint violation)"},
        422: {"description": "Error parsing the molecular structure"},
    },
)
async def HOSE_Codes(
    smiles: Annotated[
        str,
        Query(
            description="SMILES string representing the molecular structure",
            example="CCCC1CC1",
            examples=[
                "CCCC1CC1",
                "c1ccccc1",
                "CC(=O)O",
                "CCO",
                "C1CCCCC1",
                "CC(=O)Oc1ccccc1C(=O)O",
            ],
        ),
    ],
    framework: Annotated[
        str,
        Query(
            enum=["cdk", "rdkit"],
            description="Cheminformatics framework to use for HOSE code generation",
        ),
    ] = "cdk",
    spheres: Annotated[
        int,
        Query(
            description="Number of spheres (bond distance) to consider around each atom",
            ge=1,
            le=10,
        ),
    ] = 3,
    usestereo: Annotated[
        bool,
        Query(
            description="Whether to include stereochemistry information in HOSE codes (CDK only)",
        ),
    ] = False,
) -> list[str]:
    """
    ## Generate HOSE codes for a given molecule

    Generates HOSE (Hierarchically Ordered Spherical Environment) codes based on
    each atom in the given molecule. These codes are widely used in NMR chemical
    shift prediction.

    ### Parameters
    - **smiles**: A valid SMILES string (e.g. `CCCC1CC1`)
    - **framework**: Choose `cdk` (default) or `rdkit`
    - **spheres**: Number of bond spheres to encode (default: 3)
    - **usestereo**: Include stereochemistry in codes (CDK only, default: false)

    ### Returns
    An array of HOSE code strings, one for each atom in the molecule.
    """
    try:
        if framework == "cdk":
            return await getCDKHOSECodes(smiles, spheres, usestereo)
        elif framework == "rdkit":
            return await getRDKitHOSECodes(smiles, spheres)
    except UniqueViolation:
        raise HTTPException(
            status_code=409,
            detail="Unique constraint violation. Molecule already exists.",
            headers={
                "X-Error": "Molecule already exists. Duplicate entries are not allowed"
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail="Error parsing the structure " + e.message,
            headers={"X-Error": "RDKit molecule input parse error"},
        )
