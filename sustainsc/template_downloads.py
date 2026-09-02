"""Immutable workbook assets offered by the Streamlit import workflow."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class WorkbookTemplate:
    filename: str
    sha256: str


DPP_TEMPLATE = WorkbookTemplate(
    filename="SustainSCM_DPP_Traceability_Input_Template.xlsx",
    sha256="494d79798d1c6ab9079a67eb8678dc8868a98bf8e898391398dc15e3f49d087a",
)
MRV_TEMPLATE = WorkbookTemplate(
    filename="SustainSCM_MRV_Causal_Completion_Template_FINAL_BOUNDARY_RECONCILED_VALIDATED.xlsx",
    sha256="ce617a445492ce04130ef55f72a63b01accdf4f6f5dbddea4cadf5205b93739f",
)

TEMPLATE_DIRECTORY = Path(__file__).resolve().parent.parent / "templates"
XLSX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@lru_cache(maxsize=2)
def load_template_bytes(template: WorkbookTemplate) -> bytes:
    """Load a known template and reject a missing or unexpectedly changed asset."""
    path = (TEMPLATE_DIRECTORY / template.filename).resolve()
    if path.parent != TEMPLATE_DIRECTORY.resolve():
        raise ValueError(f"Template path escapes the asset directory: {template.filename}")
    payload = path.read_bytes()
    digest = sha256(payload).hexdigest()
    if digest != template.sha256:
        raise ValueError(
            f"Template checksum mismatch for {template.filename}: {digest}"
        )
    return payload
