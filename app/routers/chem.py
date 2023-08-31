from typing import Annotated
from psycopg2.errors import UniqueViolation
from app.modules.cdkmodules import getCDKHOSECodes
from fastapi import APIRouter, HTTPException, status, Query, Body
from app.modules.rdkitmodules import getRDKitHOSECodes
from app.schemas import HealthCheck
from app.schemas.alatis import AlatisModel
import requests

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
    summary="Generates HOSE codes of molecule",
    response_model=list[str],
    response_description="Returns an array of hose codes generated",
    status_code=status.HTTP_200_OK,
)
async def HOSE_Codes(
    smiles: Annotated[str, Query(examples=["CCCC1CC1"])],
    framework: Annotated[str, Query(enum=["cdk", "rdkit"])] = "cdk",
    spheres: Annotated[int, Query()] = 3,
    usestereo: Annotated[bool, Query()] = False,
) -> list[str]:
    """
    ## Generates HOSE codes for a given molecule
    Endpoint to generate HOSE codes based on each atom in the given molecule.

    Returns:
        HOSE Codes: An array of hose codes generated
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


@router.post(
    "/label-atoms",
    tags=["chem"],
    summary="Label atoms using ALATIS naming system",
    response_model=AlatisModel,
    response_description="",
    status_code=status.HTTP_200_OK,
)
async def label_atoms(
    data: Annotated[
        str,
        Body(embed=False, media_type="text/plain"),
    ]
):
    """
    ## Generates atom labels for a given molecule

    Returns:
        JSON with various representations
    """
    try:
        url = "http://alatis.nmrfam.wisc.edu/upload"
        payload = {"input_text": data, "format": "format_", "response_type": "json"}
        response = requests.request("POST", url, data=payload)
        return response.json()
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail="Error parsing the structure " + e.message,
            headers={"X-Error": "RDKit molecule input parse error"},
        )
