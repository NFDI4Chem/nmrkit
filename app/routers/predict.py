import subprocess
from fastapi import APIRouter, HTTPException, status, Response, Body
from app.schemas import HealthCheck
from urllib.parse import unquote
from app.schemas.respredict_response_schema import ResPredictModel
from app.schemas.error import ErrorResponse, BadRequestModel, NotFoundModel
import json
import uuid
from typing import Annotated
import os

router = APIRouter(
    prefix="/predict",
    tags=["predict"],
    dependencies=[],
    responses={404: {"description": "Not Found"}},
)


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
    "/respredict",
    summary="",
    responses={
        200: {
            "description": "Successful response",
            "model": ResPredictModel,
        },
        400: {"description": "Bad Request", "model": BadRequestModel},
        404: {"description": "Not Found", "model": NotFoundModel},
        422: {"description": "Unprocessable Entity", "model": ErrorResponse},
    },
)
async def predict_mol(
    data: Annotated[
        str,
        Body(
            embed=False,
            media_type="text/plain",
            openapi_examples={
                "example1": {
                    "summary": "Example: C",
                    "value": """
  CDK     09012310592D

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END""",
                },
            },
        ),
    ]
):
    """
    Standardize molblock using the ChEMBL curation pipeline
    and return the standardized molecule, SMILES, InChI, and InCHI-Key.

    Parameters:
    - **molblock**: The request body containing the "molblock" string representing the molecule to be standardized.

    Returns:
    - dict: A dictionary containing the following keys:
        - "standardized_mol" (str): The standardized molblock of the molecule.
        - "canonical_smiles" (str): The canonical SMILES representation of the molecule.
        - "inchi" (str): The InChI representation of the molecule.
        - "inchikey" (str): The InChI-Key of the molecule.

    Raises:
    - ValueError: If the SMILES string is not provided or is invalid.

    """
    try:
        if data:
            file_name = "/shared/" + str(uuid.uuid4()) + ".mol"
            f = open(file_name, "a")
            f.write(data)
            f.close()
            process = subprocess.Popen(
                [
                    "docker exec nmr-respredict python3 predict_standalone.py --filename "
                    + file_name
                ],
                stdout=subprocess.PIPE,
                shell=True,
            )
            (output, err) = process.communicate()
            process.wait()
            if err:
                raise HTTPException(status_code=500, detail=err)
            else:
                if os.path.exists(file_name):
                    os.remove(file_name)
                return Response(content=output, media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# @router.get(
#     "/predict",
#     tags=["predict"],
#     summary="Load and convert NMR raw data",
#     #response_model=List[int],
#     response_description="Load and convert NMR raw data",
#     status_code=status.HTTP_200_OK,
# )
# async def nmr_respredict(url: str):
#     """
#     ## Return nmrium json

#     Returns:
#         Return nmrium json
#     """
#     file = "/shared" + str(uuid.uuid4()) + ".mol"

#     process = subprocess.Popen(
#         ["docker exec nmr-respredict python3 ./predict_standalone.py --filename " + file],
#         stdout=subprocess.PIPE,
#         shell=True,
#     )
#     (output, err) = process.communicate()
#     process.wait()
#     if err:
#         raise HTTPException(status_code=500, detail=err)
#     else:
#         return Response(content=output, media_type="application/json")
