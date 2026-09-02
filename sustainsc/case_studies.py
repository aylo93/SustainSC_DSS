"""Curated, runnable case-study examples for the import landing page."""

from __future__ import annotations

from dataclasses import dataclass

from .template_downloads import (
    CUBA_DPP_EXAMPLE,
    CUBA_MRV_EXAMPLE,
    ROMANIA_DPP_EXAMPLE,
    ROMANIA_MRV_EXAMPLE,
    WorkbookTemplate,
)


@dataclass(frozen=True)
class CaseStudy:
    slug: str
    flag: str
    title: str
    industry: str
    location: str
    description: str
    scenario_summary: str
    dpp_summary: str
    mrv_workbook: WorkbookTemplate
    dpp_workbook: WorkbookTemplate


CASE_STUDIES = (
    CaseStudy(
        slug="cuba",
        flag="🇨🇺",
        title="Cuba",
        industry="Aggregates supply chain",
        location="Holguín Province multi-site network",
        description=(
            "Explore sustainable production and distribution of processed aggregates "
            "across VSM-C, MILP, DES and system-dynamics scenarios."
        ),
        scenario_summary="24 MRV scenarios · 0 critical completion failures",
        dpp_summary="18 product batches · 24 traceability events",
        mrv_workbook=CUBA_MRV_EXAMPLE,
        dpp_workbook=CUBA_DPP_EXAMPLE,
    ),
    CaseStudy(
        slug="romania",
        flag="🇷🇴",
        title="Romania · REEL–PLANTEC",
        industry="Cable-assembly manufacturing",
        location="Codlea, Brașov County",
        description=(
            "Run an industrial cable-assembly case with operational, logistics, social "
            "and technology indicators across four analytical methods."
        ),
        scenario_summary="18 MRV scenarios · 0 critical completion failures",
        dpp_summary="13 product batches · 26 traceability events",
        mrv_workbook=ROMANIA_MRV_EXAMPLE,
        dpp_workbook=ROMANIA_DPP_EXAMPLE,
    ),
)
