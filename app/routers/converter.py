import subprocess
from fastapi import APIRouter, HTTPException, status
from app.schemas import HealthCheck
from urllib.parse import unquote


router = APIRouter(
    prefix="/convert",
    tags=["converter"],
    dependencies=[],
    responses={404: {"description": "Not Found"}},
)

@router.get("/", include_in_schema=False)
@router.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check on Converter Module",
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


@router.get(
    "/nmrLoadSave",
    tags=["converter"],
    summary="Conversion through nmr-load save",
   # response_model=List[int],
    response_description="Conversion through nmr-load save",
    status_code=status.HTTP_200_OK,
)
async def nmr_load_save(url: str):
    """
    ## Return nmr_load save result

    Returns:
        Return nmr_load save result
    """
   # url = "https://cheminfo.github.io/bruker-data-test/data/zipped/aspirin-1h.zip"
   # command = f"docker exec -it nmrkit_nmr-load-save_1 nmr-cli -u {url}"
    try:
        process = subprocess.run("docker exec -it nmrkit_nmr-load-save_1 nmr-cli -u " + unqoute(url),
                                stdout=subprocess.PIPE,
                                capture_output=True,
                                shell=True)
       # process = subprocess.run(['docker', 'exec', '-it', 'nmrkit_nmr-load-save_1', 'nmr-cli', '-u', unquote(url)], capture_output=True, shell=True)
       # print('printing result..')
       # print(process.stdout)
       # return {"output": process.stdout}
       # (output, err) = process.communicate()
       # process_status = process.wait()
       # print (output)
       # return str(output)
        while True:
            output = process.stdout.readline()
            print(output.strip())
            # Do something else
            return_code = process.poll()
            if return_code is not None:
                print('RETURN CODE', return_code)
                # Process has finished, read rest of the output
                for output in process.stdout.readlines():
                    print(output.strip())
                break
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error" + e.message)
