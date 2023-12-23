import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


@pytest.fixture
def test_smiles():
    return "CC(=O)C"


@pytest.fixture
def molfile():
    return """
  CDK     08302311362D

  4  3  0  0  0  0  0  0  0  0999 V2000
    0.9743    0.5625    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3248    1.3125    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3248    2.8125    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -1.6238    0.5625    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  2  0  0  0  0
  2  4  1  0  0  0  0
M  END
"""


def test_chem_index():
    response = client.get("/latest/chem/")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


@pytest.mark.parametrize(
    "smiles, boolean, framework, expected_result",
    [
        (
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            False,
            "cdk",
            '["C-4;N(CC/=CC,=N/)","N-3;CCC(=CC,=N,/N&,=ON,&/)","C-3;=NN(C,CC/=&N,=&C,/)","N-2;=CC(N,=CN/&C,C&,CC/)","C-3;=CNN(CN,CC,=C/=ON,&C,=O&,,&/)","C-3;=CCN(NN,=ON,CC/CC,=&,,&C,=&,/)","C-3;=OCN(,=CN,CC/NN,CC,=O&,/)","O-1;=C(CN/=CN,CC/)","N-3;CCC(=OC,=ON,/,=CN,,&C/)","C-3;=ONN(,CC,CC/=OC,,=&N,/)","O-1;=C(NN/CC,CC/)","N-3;CCC(=CN,=ON,/CN,=C,,&C/)","C-4;N(CC/=CN,=ON/)","C-4;N(CC/=OC,=ON/)"]',
        ),
        (
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            True,
            "rdkit",
            '["C-4;N(*C*C/*C*C,*N/)","N-3;*C*CC(*C*C,*N,/*N*&,=O*N,*&/)","C-3;*N*N(*CC,*C/*C*&,,*&*N/)","N-2;*C*C(*C*N,*N/*C*&,*CC,*&C/)","C-3;*C*N*N(*C*N,*CC,*C/=O*N,*&C,=O*&,,*&/)","C-3;*C*C*N(*N*N,=O*N,*CC/*CC,*&,,*&C,*&,/)","C-3;=O*C*N(,*C*N,*CC/*N*N,*CC,=O*&,/)","O-1;=C(*C*N/*C*N,*CC/)","N-3;*C*CC(=O,=O*C,*N,/,,*C*N,*&C/)","C-3;=O*N*N(,*C,*CC,C/*C*N,=O*&,,/)","O-1;=C(*N*N/*C,*C,C,C/)","N-3;*C*CC(*C*N,=O*N,/*C*N,*C,,*&C/)","C-4;N(*C*C/*C*N,=O*N/)"]',
        ),
        (
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            False,
            "rdkit",
            '["C-4;N(*C*C/*C*C,*N/)","N-3;*C*CC(*C*C,*N,/*N*&,=O*N,*&/)","C-3;*N*N(*CC,*C/*C*&,,*&*N/)","N-2;*C*C(*C*N,*N/*C*&,*CC,*&C/)","C-3;*C*N*N(*C*N,*CC,*C/=O*N,*&C,=O*&,,*&/)","C-3;*C*C*N(*N*N,=O*N,*CC/*CC,*&,,*&C,*&,/)","C-3;=O*C*N(,*C*N,*CC/*N*N,*CC,=O*&,/)","O-1;=C(*C*N/*C*N,*CC/)","N-3;*C*CC(=O,=O*C,*N,/,,*C*N,*&C/)","C-3;=O*N*N(,*C,*CC,C/*C*N,=O*&,,/)","O-1;=C(*N*N/*C,*C,C,C/)","N-3;*C*CC(*C*N,=O*N,/*C*N,*C,,*&C/)","C-4;N(*C*C/*C*N,=O*N/)"]',
        ),
        (
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            True,
            "cdk",
            '["C-4;N(CC/=CC,=N/)","N-3-5;CCC(=CC,=N,/N&,=ON,&/)","C-3-5;=NN(C,CC/=&N,=&C,/)","N-2-5;=CC(N,=CN/&C,C&,CC/)","C-3-56;=CNN(CN,CC,=C/=ON,&C,=O&,,&/)","C-3-56;=CCN(NN,=ON,CC/CC,=&,,&C,=&,/)","C-3-6;=OCN(,=CN,CC/NN,CC,=O&,/)","O-1;=C(CN/=CN,CC/)","N-3-6;CCC(=OC,=ON,/,=CN,,&C/)","C-3-6;=ONN(,CC,CC/=OC,,=&N,/)","O-1;=C(NN/CC,CC/)","N-3-6;CCC(=CN,=ON,/CN,=C,,&C/)","C-4;N(CC/=CN,=ON/)","C-4;N(CC/=OC,=ON/)"]',
        ),
    ],
)
def test_hosecode(smiles, boolean, framework, expected_result):
    response = client.get(
        f"/latest/chem/hosecode?smiles={smiles}&framework={framework}&spheres=3&usestereo={boolean}"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.text == expected_result


def test_label_atoms(molfile):
    response = client.post(
        "/latest/chem/label-atoms", data=molfile, headers={"Content-Type": "text/plain"}
    )
    assert response.status_code == 200
    assert "html_url" in response.json()
    assert "inchi" in response.json()
    assert "key" in response.json()
    assert "status" in response.json()
    assert "structure" in response.json()


# Run the tests
if __name__ == "__main__":
    pytest.main()
