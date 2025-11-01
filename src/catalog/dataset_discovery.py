"""Automatic dataset discovery from data.gov.in."""

import requests
from typing import List, Dict, Optional
import logging
import google.generativeai as genai

from src.config import DATA_GOV_API_KEY, GEMINI_API_KEY
from src.catalog.dataset_catalog import DatasetCatalog, DatasetMetadata
from src.catalog.seed_datasets import is_authorized_publisher, get_authorized_publishers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDiscovery:
    """Automatically discover and categorize datasets from data.gov.in."""

    def __init__(self, api_key: str = DATA_GOV_API_KEY):
        self.api_key = api_key
        self.catalog = DatasetCatalog()
        genai.configure(api_key=GEMINI_API_KEY)
        self.llm = genai.GenerativeModel('gemini-2.0-flash-lite')

        # Authorized publishers
        self.authorized_publishers = get_authorized_publishers()
        logger.info(f"Authorized publishers: {', '.join(self.authorized_publishers)}")

        # Search terms for different categories
        self.search_terms = {
            "climate": [
                "rainfall", "IMD rainfall", "temperature", "weather",
                "meteorological", "precipitation", "monsoon"
            ],
            "agriculture": [
                "crop production", "agricultural statistics", "agriculture",
                "crop yield", "farming", "harvest", "cultivation"
            ]
        }

    def search_datasets(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search for datasets on data.gov.in.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of dataset metadata
        """
        try:
            # data.gov.in search API endpoint - using /lists which works for catalog search
            search_url = "https://api.data.gov.in/lists"

            params = {
                "api-key": self.api_key,
                "format": "json",
                "filters[title]": query,
                "limit": max_results
            }

            logger.info(f"Searching data.gov.in for: {query}")
            response = requests.get(search_url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if "records" in data:
                    return data["records"]
                elif "result" in data and "records" in data["result"]:
                    return data["result"]["records"]
                else:
                    logger.warning(f"Unexpected response structure from data.gov.in")
                    return []
            else:
                logger.error(f"Search failed with status {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error searching datasets: {e}")
            return []

    def categorize_dataset(self, dataset: Dict) -> Optional[str]:
        """
        Use LLM to categorize a dataset as climate or agriculture.

        Args:
            dataset: Dataset metadata dictionary

        Returns:
            Category string ("climate" or "agriculture") or None
        """
        title = dataset.get("title", "")
        description = dataset.get("desc", dataset.get("description", ""))

        prompt = f"""Categorize this dataset as either "climate" or "agriculture" based on its title and description.

Title: {title}
Description: {description}

Rules:
- Return ONLY one word: "climate" or "agriculture" or "other"
- Climate datasets include: rainfall, temperature, weather, meteorological data, IMD data
- Agriculture datasets include: crop production, yield, farming, agricultural statistics
- If it doesn't clearly fit either category, return "other"

Category:"""

        try:
            response = self.llm.generate_content(prompt)
            category = response.text.strip().lower()

            if category in ["climate", "agriculture"]:
                return category
            else:
                return None

        except Exception as e:
            logger.error(f"Error categorizing dataset: {e}")
            return None

    def discover_all_datasets(self) -> Dict[str, List[DatasetMetadata]]:
        """
        Discover datasets for all categories.

        Returns:
            Dictionary mapping category to list of dataset metadata
        """
        discovered = {"climate": [], "agriculture": []}

        for category, search_terms in self.search_terms.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Discovering {category} datasets...")
            logger.info(f"{'='*60}")

            all_datasets = []

            # Search using each term
            for term in search_terms:
                datasets = self.search_datasets(term, max_results=5)
                all_datasets.extend(datasets)

            # Remove duplicates based on resource_id
            seen = set()
            unique_datasets = []
            for ds in all_datasets:
                resource_id = self._extract_resource_id(ds)
                if resource_id and resource_id not in seen:
                    seen.add(resource_id)
                    unique_datasets.append(ds)

            logger.info(f"Found {len(unique_datasets)} unique datasets for {category}")

            # Convert to DatasetMetadata (filtering by authorized publishers happens inside)
            for dataset in unique_datasets[:10]:  # Limit to top 10 per category
                metadata = self._convert_to_metadata(dataset, category)
                if metadata:
                    discovered[category].append(metadata)
                    # Add to catalog
                    self.catalog.add_dataset(metadata)

            logger.info(f"Added {len(discovered[category])} {category} datasets from authorized publishers to catalog")

        return discovered

    def _extract_resource_id(self, dataset: Dict) -> Optional[str]:
        """Extract resource ID from dataset metadata."""
        # Try different possible fields - index_name is the primary field from /lists endpoint
        resource_id = (
            dataset.get("index_name") or
            dataset.get("resource_id") or
            dataset.get("id") or
            dataset.get("org_id")
        )
        return resource_id

    def _convert_to_metadata(self, dataset: Dict, category: str) -> Optional[DatasetMetadata]:
        """Convert data.gov.in dataset to DatasetMetadata."""
        try:
            resource_id = self._extract_resource_id(dataset)
            if not resource_id:
                logger.warning(f"No resource ID found for dataset: {dataset.get('title', 'Unknown')}")
                return None

            # Use exact title from data.gov.in
            title = dataset.get("title", "Unknown Dataset")
            description = dataset.get("desc", dataset.get("description", ""))

            # Extract publisher/ministry
            publisher = (
                dataset.get("org", {}).get("title") if isinstance(dataset.get("org"), dict)
                else dataset.get("source", "data.gov.in")
            )

            # FILTER: Only allow authorized publishers
            if not is_authorized_publisher(publisher):
                logger.info(f"Skipping dataset '{title}' from unauthorized publisher: {publisher}")
                return None

            # Create unique, descriptive name
            # If title is very generic (single word), append publisher
            name = title
            if publisher and publisher != "data.gov.in":
                # Check if title is generic or short
                if len(title.split()) <= 2 or title.lower() in ['rainfall', 'temperature', 'production', 'crop', 'data']:
                    # Append publisher for clarity
                    name = f"{title} ({publisher})"

            # Determine format
            format_type = dataset.get("format", "json").lower()
            if format_type not in ["json", "csv", "xlsx", "xls"]:
                format_type = "json"  # Default to json

            # Get sample columns if available
            fields = dataset.get("field", [])
            if isinstance(fields, list):
                sample_columns = ",".join([f.get("id", "") for f in fields[:10]])
            else:
                sample_columns = ""

            metadata = DatasetMetadata(
                dataset_id=resource_id,
                resource_id=resource_id,
                name=name,  # Using enhanced unique name
                publisher=publisher,
                format=format_type,
                category=category,
                sample_columns=sample_columns,
                last_updated=dataset.get("updated_date")
            )

            return metadata

        except Exception as e:
            logger.error(f"Error converting dataset to metadata: {e}")
            return None

    def discover_and_add_datasets_for_question(self, question: str) -> List[Dict]:
        """
        Discover and add new datasets on-demand based on the user's question.

        This method:
        1. Extracts search keywords from the question using LLM
        2. Searches data.gov.in for relevant datasets
        3. Adds new datasets to the catalog if not already present
        4. Returns the list of relevant dataset IDs (both existing and newly added)

        Args:
            question: User's natural language question

        Returns:
            List of dataset metadata dictionaries
        """
        logger.info(f"Discovering datasets for question: {question}")

        # Step 1: Extract search keywords from the question using LLM
        keywords = self._extract_search_keywords(question)
        logger.info(f"Extracted keywords: {keywords}")

        # Step 2: Search for datasets using these keywords
        discovered_datasets = []
        for keyword in keywords:
            datasets = self.search_datasets(keyword, max_results=5)
            discovered_datasets.extend(datasets)

        # Remove duplicates
        seen = set()
        unique_datasets = []
        for ds in discovered_datasets:
            resource_id = self._extract_resource_id(ds)
            if resource_id and resource_id not in seen:
                seen.add(resource_id)
                unique_datasets.append(ds)

        logger.info(f"Found {len(unique_datasets)} unique datasets from search")

        # Step 3: Add new datasets to catalog (skip if already present or unauthorized)
        newly_added = []
        skipped_unauthorized = 0

        for dataset in unique_datasets:
            resource_id = self._extract_resource_id(dataset)

            # Check if dataset already exists in catalog
            existing = self.catalog.get_dataset(resource_id)
            if existing:
                logger.info(f"Dataset {resource_id} already in catalog, skipping")
                continue

            # Check publisher BEFORE calling LLM (save API calls)
            publisher = (
                dataset.get("org", {}).get("title") if isinstance(dataset.get("org"), dict)
                else dataset.get("source", "data.gov.in")
            )

            if not is_authorized_publisher(publisher):
                skipped_unauthorized += 1
                logger.debug(f"Skipping dataset {resource_id} from unauthorized publisher: {publisher}")
                continue

            # Determine category using LLM
            category = self.categorize_dataset(dataset)
            if not category:
                logger.info(f"Dataset {resource_id} doesn't fit climate/agriculture categories, skipping")
                continue

            # Convert to metadata and add to catalog
            metadata = self._convert_to_metadata(dataset, category)
            if metadata:
                self.catalog.add_dataset(metadata)
                newly_added.append(metadata)
                logger.info(f"Added new dataset: {metadata.name} ({resource_id})")

        logger.info(f"Added {len(newly_added)} new datasets from authorized publishers to catalog")
        if skipped_unauthorized > 0:
            logger.info(f"Skipped {skipped_unauthorized} datasets from unauthorized publishers")

        # Step 4: Return all relevant datasets (both existing and new)
        return self.get_relevant_datasets(question)

    def _extract_search_keywords(self, question: str) -> List[str]:
        """
        Extract search keywords from a user question using LLM.

        Args:
            question: User's natural language question

        Returns:
            List of search keywords
        """
        prompt = f"""Extract 2-4 search keywords from this question that would be useful for finding relevant datasets on data.gov.in.

User Question: {question}

Rules:
- Return keywords related to the topic (e.g., "rainfall", "crop production", "temperature")
- Focus on data types, not locations or time periods
- Return ONLY a comma-separated list of keywords
- Examples: "rainfall, monsoon" or "crop yield, agriculture production"

Search Keywords:"""

        try:
            response = self.llm.generate_content(prompt)
            keywords_str = response.text.strip()
            keywords = [k.strip() for k in keywords_str.split(",")]
            return keywords[:4]  # Limit to 4 keywords max
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            # Fallback: basic keyword extraction
            return self._basic_keyword_extraction(question)

    def _basic_keyword_extraction(self, question: str) -> List[str]:
        """Fallback method for keyword extraction without LLM."""
        question_lower = question.lower()
        keywords = []

        # Climate-related keywords
        climate_terms = ["rainfall", "temperature", "weather", "monsoon", "precipitation", "climate"]
        for term in climate_terms:
            if term in question_lower:
                keywords.append(term)

        # Agriculture-related keywords
        agri_terms = ["crop", "agriculture", "farming", "yield", "production", "harvest"]
        for term in agri_terms:
            if term in question_lower:
                keywords.append(term)

        return keywords[:4] if keywords else ["rainfall", "crop production"]

    def get_relevant_datasets(self, question: str) -> List[str]:
        """
        Get relevant dataset IDs for a user question using LLM.

        Args:
            question: User's natural language question

        Returns:
            List of relevant dataset resource IDs
        """
        # Get all cataloged datasets
        climate_datasets = self.catalog.list_datasets(category="climate")
        agri_datasets = self.catalog.list_datasets(category="agriculture")

        # Build dataset list for LLM
        dataset_list = []
        for ds in climate_datasets + agri_datasets:
            dataset_list.append(f"- {ds['dataset_id']}: {ds['name']} ({ds['category']})")

        dataset_str = "\n".join(dataset_list)

        prompt = f"""Given the following user question and available datasets, select the most relevant dataset IDs that would help answer the question.

User Question: {question}

Available Datasets:
{dataset_str}

Return ONLY a comma-separated list of dataset IDs that are relevant.
If the question involves both climate and agriculture, include datasets from both categories.
If no datasets are relevant, return "NONE".

Relevant Dataset IDs:"""

        try:
            response = self.llm.generate_content(prompt)
            result = response.text.strip()

            if result == "NONE":
                return []

            # Parse comma-separated IDs
            dataset_ids = [id.strip() for id in result.split(",")]
            return dataset_ids

        except Exception as e:
            logger.error(f"Error getting relevant datasets: {e}")
            # Fallback: return all datasets
            return [ds["dataset_id"] for ds in climate_datasets + agri_datasets]
