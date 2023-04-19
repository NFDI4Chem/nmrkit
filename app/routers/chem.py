from fastapi import APIRouter
from app.modules.cdkmodules import getCDKHOSECodes
from app.modules.rdkitmodules import getRDKitHOSECodes

router = APIRouter(
    prefix="/chem",
    tags=["chem"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def chem_index():
    return {"module": "chem", "message": "Successful", "status": 200}


@router.get("/hosecode")
async def HOSE_Codes(framework: str, smiles: str, spheres: int, ringsize: bool = False):
    if smiles:
        if framework == 'cdk':
            return await getCDKHOSECodes(smiles, spheres, ringsize)
        elif framework == 'rdkit':
            return await getRDKitHOSECodes(smiles, spheres)
    else:
        return "Error reading SMILES string, check again."