"""Pre-seeded dataset resource IDs from data.gov.in for reliable startup."""

from typing import List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Authorized publishers - only fetch data from these trusted sources
AUTHORIZED_PUBLISHERS = [
    "India Meteorological Department",
    "Ministry of Agriculture & Farmers Welfare",
    "Ministry of Agriculture and Farmers Welfare",  # Alternative spelling
    "IMD",  # Abbreviation
]


@dataclass
class SeedDataset:
    """Seed dataset configuration."""
    dataset_id: str
    resource_id: str
    name: str
    publisher: str
    format: str
    category: str
    sample_columns: str = ""


# Known working resource IDs from data.gov.in
# These datasets are reliable and provide good coverage for climate and agriculture queries
SEED_DATASETS: List[SeedDataset] = [
    # Climate Datasets
    SeedDataset(
        dataset_id="9ef84268-d588-465a-a308-a864a43d0070",
        resource_id="9ef84268-d588-465a-a308-a864a43d0070",
        name="Rainfall Statistics",
        publisher="India Meteorological Department",
        format="json",
        category="climate",
        sample_columns="state_name,district,year,jan,feb,mar,apr,may,jun,jul,aug,sep,oct,nov,dec,annual"
    ),
    SeedDataset(
        dataset_id="d13f0cc6-efc0-4da3-aadc-a01532c29b34",
        resource_id="d13f0cc6-efc0-4da3-aadc-a01532c29b34",
        name="District-wise Seasonal Rainfall",
        publisher="India Meteorological Department",
        format="json",
        category="climate",
        sample_columns="state_ut_name,district,year,subdivision,annual,jan_feb,mar_may,jun_sep,oct_dec"
    ),
    SeedDataset(
        dataset_id="d6e6973a-c4cf-4d35-86e0-b9ea0e45a496",
        resource_id="d6e6973a-c4cf-4d35-86e0-b9ea0e45a496",
        name="Actual Rainfall - State Wise",
        publisher="Ministry of Agriculture & Farmers Welfare",
        format="json",
        category="climate",
        sample_columns="state_name,year,subdivision,actual_rainfall,normal_rainfall,deviation"
    ),

    # Agriculture Datasets
    SeedDataset(
        dataset_id="52e9a4e9-d74e-4ca7-9cfa-23c3e5799c4f",
        resource_id="52e9a4e9-d74e-4ca7-9cfa-23c3e5799c4f",
        name="Crop Production Statistics",
        publisher="Ministry of Agriculture & Farmers Welfare",
        format="json",
        category="agriculture",
        sample_columns="state_name,district,crop_year,season,crop,area,production,yield"
    ),
    SeedDataset(
        dataset_id="d6fb2a81-97c4-4e65-b4b4-bf0e7b8f5e4d",
        resource_id="d6fb2a81-97c4-4e65-b4b4-bf0e7b8f5e4d",
        name="District-wise Crop Production",
        publisher="Ministry of Agriculture & Farmers Welfare",
        format="json",
        category="agriculture",
        sample_columns="state,district,crop,year,area_in_hectare,production_in_tonnes,yield_per_hectare"
    ),
    SeedDataset(
        dataset_id="94eee5ae-8b3b-4f27-bc61-de7f926c7f51",
        resource_id="94eee5ae-8b3b-4f27-bc61-de7f926c7f51",
        name="State-wise Agricultural Production",
        publisher="Ministry of Agriculture & Farmers Welfare",
        format="json",
        category="agriculture",
        sample_columns="state_name,year,crop,production,area,productivity"
    ),
    SeedDataset(
        dataset_id="6a1eb413-e926-4810-92c6-f9cf14074062",
        resource_id="6a1eb413-e926-4810-92c6-f9cf14074062",
        name="Foodgrains Production",
        publisher="Ministry of Agriculture & Farmers Welfare",
        format="json",
        category="agriculture",
        sample_columns="year,state,foodgrain,area,production,yield"
    ),
]


def get_seed_datasets() -> List[SeedDataset]:
    """Get list of seed datasets to pre-populate the catalog."""
    return SEED_DATASETS


def get_seed_count() -> dict:
    """Get count of seed datasets by category."""
    climate_count = len([d for d in SEED_DATASETS if d.category == "climate"])
    agriculture_count = len([d for d in SEED_DATASETS if d.category == "agriculture"])

    return {
        "total": len(SEED_DATASETS),
        "climate": climate_count,
        "agriculture": agriculture_count
    }


def is_authorized_publisher(publisher: str) -> bool:
    """
    Check if a publisher is in the authorized list.

    Args:
        publisher: Publisher name to check

    Returns:
        True if publisher is authorized, False otherwise
    """
    if not publisher:
        return False

    publisher_lower = publisher.lower().strip()

    for authorized in AUTHORIZED_PUBLISHERS:
        if authorized.lower() in publisher_lower or publisher_lower in authorized.lower():
            return True

    return False


def get_authorized_publishers() -> List[str]:
    """Get list of authorized publishers."""
    return AUTHORIZED_PUBLISHERS.copy()
