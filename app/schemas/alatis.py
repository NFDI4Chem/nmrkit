from __future__ import annotations

from pydantic import BaseModel


class AlatisModel(BaseModel):
    html_url: str
    inchi: str
    key: str
    status: str
    structure: str
