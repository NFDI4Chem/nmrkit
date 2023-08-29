import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


@pytest.fixture
def test_smiles():
    return "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


def test_chem_index():
    response = client.get("/latest/chem/")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

# Run the tests
if __name__ == "__main__":
    pytest.main()