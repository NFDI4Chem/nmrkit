import json
import subprocess

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import validate

client = TestClient(app)

MOLFILE = """
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

REPORT = {
    "engine": {"name": "nmrshift", "source": "nmrshiftdb2 quickcheck", "url": "https://example.test"},
    "solvent": "Chloroform-D1 (CDCl3)",
    "verdict": "accept",
    "reports": {"13C": {"mark": 10, "result": "accept"}},
    "assignment_check": {"result": "consistent", "rows": [], "suggestions": [], "issues": []},
    "adjustments": [],
}


def acetone_request():
    return {
        "structure": {"molfile": MOLFILE, "source": "manual"},
        "conditions": {"solvent": "CDCl3"},
        "assignments": [
            {"nucleus": "13C", "atoms": [1, 4], "label": "CH3", "shift": 30.8},
            {"nucleus": "13C", "atoms": [2], "label": "C=O", "shift": 206.7},
            {"nucleus": "1H", "atoms": [1, 4], "label": "CH3", "shift": 2.17, "n_h": 6},
        ],
    }


class FakeCli:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append({"cmd": cmd, **kwargs})
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, self.stderr)


@pytest.fixture(autouse=True)
def empty_cache():
    validate.report_cache.clear()
    yield
    validate.report_cache.clear()


def install_cli(monkeypatch, fake):
    monkeypatch.setattr(validate.subprocess, "run", fake)
    return fake


def test_validate_index():
    response = client.get("/latest/validate/")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_assignments_are_piped_to_nmr_cli_and_the_report_is_returned(monkeypatch):
    fake = install_cli(monkeypatch, FakeCli(stdout=json.dumps(REPORT).encode()))

    response = client.post("/latest/validate/assignments", json=acetone_request())

    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "accept"
    assert body["cached"] is False

    call = fake.calls[0]
    assert call["cmd"] == ["docker", "exec", "-i", "nmr-converter", "nmr-cli", "validate-assignments"]
    sent = json.loads(call["input"].decode())
    assert sent["structure"]["molfile"] == MOLFILE
    assert sent["assignments"][2] == {"nucleus": "1H", "atoms": [1, 4], "label": "CH3", "shift": 2.17, "n_h": 6}


def test_identical_requests_are_served_from_cache(monkeypatch):
    fake = install_cli(monkeypatch, FakeCli(stdout=json.dumps(REPORT).encode()))

    client.post("/latest/validate/assignments", json=acetone_request())
    response = client.post("/latest/validate/assignments", json=acetone_request())

    assert response.status_code == 200
    assert response.json()["cached"] is True
    assert len(fake.calls) == 1


def test_invalid_structure_from_cli_maps_to_422(monkeypatch):
    error = {"error": "invalid_input", "message": "Only V2000 molfiles are supported"}
    install_cli(monkeypatch, FakeCli(returncode=2, stderr=json.dumps(error).encode()))

    response = client.post("/latest/validate/assignments", json=acetone_request())

    assert response.status_code == 422
    assert response.json()["detail"] == error


def test_unavailable_quickcheck_maps_to_503_and_is_not_cached(monkeypatch):
    error = {"error": "quickcheck_unavailable", "message": "timeout of 60000ms exceeded"}
    fake = install_cli(monkeypatch, FakeCli(returncode=3, stderr=json.dumps(error).encode()))

    first = client.post("/latest/validate/assignments", json=acetone_request())
    second = client.post("/latest/validate/assignments", json=acetone_request())

    assert first.status_code == 503
    assert second.status_code == 503
    assert len(fake.calls) == 2


def test_request_schema_is_validated_before_calling_the_cli(monkeypatch):
    fake = install_cli(monkeypatch, FakeCli(stdout=json.dumps(REPORT).encode()))
    payload = acetone_request()
    payload["assignments"][0]["nucleus"] = "15N"

    response = client.post("/latest/validate/assignments", json=payload)

    assert response.status_code == 422
    assert fake.calls == []


def test_missing_docker_maps_to_500(monkeypatch):
    def missing_docker(cmd, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(validate.subprocess, "run", missing_docker)

    response = client.post("/latest/validate/assignments", json=acetone_request())

    assert response.status_code == 500
