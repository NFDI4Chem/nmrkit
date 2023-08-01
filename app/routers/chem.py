from typing import List
from typing import Annotated
from app.core.config import LWREG_CONFIG
from psycopg2.errors import UniqueViolation
from app.modules.cdkmodules import getCDKHOSECodes
from fastapi import APIRouter, HTTPException, Body, status, Query
from app.modules.rdkitmodules import getRDKitHOSECodes
from lwreg.utils import initdb, bulk_register, query, retrieve, RegistrationFailureReasons
from app import schemas
from typing import Annotated
from rdkit import Chem
from io import BytesIO

router = APIRouter(
    prefix="/chem",
    tags=["chem"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
)


@router.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check on Chem Module",
    response_description="Returns HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=schemas.HealthCheck,
    include_in_schema=False,
)
def get_health() -> schemas.HealthCheck:
    """
    ## Perform a Health Check
    Endpoint to perform a healthcheck on. This endpoint can primarily be used Docker
    to ensure a robust container orchestration and management is in place. Other
    services which rely on proper functioning of the API service will not deploy if this
    endpoint returns any other HTTP status code except 200 (OK).

    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    return schemas.HealthCheck(status="OK")


@router.post("/init")
async def register_compounds(confirm: Annotated[bool, Body(embed=True)] = False):
    """
    ## Initializes the registration database

    NOTE that this call destroys any existing information in the registration database

    Keyword arguments:
    confirm -- if set to False we immediately return
    """
    return initdb(config=LWREG_CONFIG, confirm=confirm)


@router.post("/register")
async def register_compounds(
    data: Annotated[
        str,
        Body(embed=False,  media_type='text/plain'),
    ] = 'CCCC'
):
    """
    ## Registers a new molecule, assuming it doesn't already exist,
    and returns the new registry number(s) (molregno)

    only one of the molecule format objects should be provided

    molblock   -- MOL or SDF block
    smiles     -- smiles
    """
    try:
        if "$$$$" in data:
            molStream = BytesIO(data.encode('utf8'))
            mols =[m for m in Chem.ForwardSDMolSupplier(molStream)]
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
    except:
        raise HTTPException(
            status_code=422,
            detail="None of the ",
        )

@router.get("/query")
async def query_compounds(smi: str):
    """
    ## Generates HOSE codes for a given molecule
    Endpoint to generate HOSE codes based on for each atom in the givem molecule.

    Returns:
        HOSE Codes: An array of hose codes generated
    """
    try:
        res = query(smiles=smi, config=LWREG_CONFIG)
        return res
    except:
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error",
        )


@router.get("/retrieve")
async def retrieve_compounds(ids: List[int]):
    """
    ## Generates HOSE codes for a given molecule
    Endpoint to generate HOSE codes based on for each atom in the givem molecule.

    Returns:
        HOSE Codes: An array of hose codes generated
    """
    try:
        res = retrieve(ids=ids, config=LWREG_CONFIG)
        return res
    except:
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error",
        )


@router.get(
    "/hosecode",
    tags=["chem"],
    summary="Generating HOSE codes of molecule",
    response_model=list[str],
    response_description="Returns HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
)
async def HOSE_Codes(
    smiles: Annotated[str, Query(examples=["A very nice Item"])],
    framework: Annotated[str, Query(enum=["cdk", "rdkit"])] = "cdk",
    spheres: Annotated[int, Query()] = None,
    usestereo: Annotated[bool, Query()] = False,
) -> list[str]:
    """
    ## Generates HOSE codes for a given molecule
    Endpoint to generate HOSE codes based on for each atom in the givem molecule.

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
    except:
        raise HTTPException(
            status_code=422,
            detail="Error paring the structure",
            headers={"X-Error": "RDKit molecule input parse error"},
        )
