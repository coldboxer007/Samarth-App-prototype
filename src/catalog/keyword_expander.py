"""Keyword expansion and permutation for enhanced dataset discovery."""

import logging
from typing import List, Set, Dict, Tuple
from itertools import combinations, product

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeywordExpander:
    """Expands keywords with synonyms, variations, and combinations for better search coverage."""

    # Domain-specific synonyms and variations
    SYNONYMS = {
        # Climate/Weather terms
        "rainfall": ["rain", "precipitation", "monsoon", "rainy", "downpour"],
        "temperature": ["temp", "thermal", "heat", "warming", "cooling"],
        "weather": ["meteorological", "climate", "atmospheric", "IMD"],
        "climate": ["weather", "meteorological", "climatic"],

        # Agriculture terms
        "crop": ["crops", "farming", "cultivation", "agricultural produce"],
        "production": ["yield", "output", "harvest", "produce"],
        "agriculture": ["farming", "agricultural", "agri", "cultivation", "agro"],
        "yield": ["production", "output", "harvest"],
        "farming": ["agriculture", "cultivation", "agri"],

        # Indian state variations (historical names)
        "odisha": ["orissa", "odisha", "orrisa"],
        "mumbai": ["bombay", "mumbai"],
        "chennai": ["madras", "chennai"],
        "kolkata": ["calcutta", "kolkata"],
        "bengaluru": ["bangalore", "bengaluru"],
        "uttarakhand": ["uttaranchal", "uttarakhand"],
        "chhattisgarh": ["chattisgarh", "chhattisgarh"],
        "telangana": ["telengana", "telangana"],

        # Time-related terms
        "annual": ["yearly", "per year", "year"],
        "monthly": ["per month", "month"],
        "seasonal": ["season", "kharif", "rabi", "zaid"],

        # Statistical terms
        "average": ["mean", "avg"],
        "total": ["sum", "aggregate"],
        "trend": ["pattern", "change", "variation"],
    }

    # Abbreviations and expansions
    ABBREVIATIONS = {
        "IMD": ["India Meteorological Department", "Indian Meteorological", "imd"],
        "MOAFW": ["Ministry of Agriculture & Farmers Welfare", "agriculture ministry"],
        "mm": ["millimeter", "millimetre"],
        "ha": ["hectare", "hectares"],
    }

    def __init__(self):
        """Initialize the keyword expander."""
        self.cache: Dict[str, List[str]] = {}

    def expand_keyword(self, keyword: str) -> List[str]:
        """
        Expand a single keyword into variations and synonyms.

        Args:
            keyword: Single keyword to expand

        Returns:
            List of keyword variations including the original
        """
        if keyword in self.cache:
            return self.cache[keyword]

        keyword_lower = keyword.lower().strip()
        expansions = {keyword_lower}  # Use set to avoid duplicates

        # Add synonyms
        if keyword_lower in self.SYNONYMS:
            expansions.update(self.SYNONYMS[keyword_lower])

        # Add abbreviations
        if keyword_lower in self.ABBREVIATIONS:
            expansions.update(self.ABBREVIATIONS[keyword_lower])

        # Check if keyword appears as synonym/abbrev of another term
        for main_term, variants in self.SYNONYMS.items():
            if keyword_lower in variants:
                expansions.add(main_term)
                expansions.update(variants)

        for main_term, variants in self.ABBREVIATIONS.items():
            if keyword_lower.upper() in [v.upper() for v in variants]:
                expansions.add(main_term)
                expansions.update(variants)

        result = list(expansions)
        self.cache[keyword] = result

        logger.debug(f"Expanded '{keyword}' to {len(result)} variations: {result[:5]}...")
        return result

    def expand_keywords(self, keywords: List[str]) -> List[str]:
        """
        Expand multiple keywords.

        Args:
            keywords: List of keywords

        Returns:
            Deduplicated list of all expanded keywords
        """
        all_expansions = set()

        for keyword in keywords:
            expansions = self.expand_keyword(keyword)
            all_expansions.update(expansions)

        return list(all_expansions)

    def generate_keyword_combinations(
        self,
        keywords: List[str],
        max_combinations: int = 20,
        include_single: bool = True,
        include_pairs: bool = True,
        expand_keywords: bool = True
    ) -> List[str]:
        """
        Generate keyword combinations for comprehensive search.

        Strategy:
        1. Expand each keyword to include synonyms/variations
        2. Use single keywords
        3. Use pairs of keywords (e.g., "rainfall" + "odisha")
        4. Prioritize most relevant combinations

        Args:
            keywords: Original keywords from user query
            max_combinations: Maximum number of search queries to generate
            include_single: Include individual keywords
            include_pairs: Include pairs of keywords
            expand_keywords: Whether to expand keywords with synonyms

        Returns:
            List of search query strings
        """
        search_queries = []

        # Step 1: Expand keywords if requested
        if expand_keywords:
            expanded_keywords = self.expand_keywords(keywords)
            logger.info(f"Expanded {len(keywords)} keywords to {len(expanded_keywords)} variations")
        else:
            expanded_keywords = keywords

        # Step 2: Add single keywords (original and expanded)
        if include_single:
            # Prioritize original keywords first
            for keyword in keywords:
                if len(search_queries) >= max_combinations:
                    break
                search_queries.append(keyword)

            # Then add expanded single keywords
            for keyword in expanded_keywords:
                if len(search_queries) >= max_combinations:
                    break
                if keyword not in search_queries:
                    search_queries.append(keyword)

        # Step 3: Add pairs of keywords (for more specific searches)
        if include_pairs and len(search_queries) < max_combinations:
            # Pair original keywords with expanded keywords
            keyword_pairs = []

            # Original keyword pairs
            if len(keywords) >= 2:
                for k1, k2 in combinations(keywords, 2):
                    keyword_pairs.append(f"{k1} {k2}")

            # Cross-product of original with expanded (most valuable combinations)
            for orig in keywords:
                orig_expansions = self.expand_keyword(orig)
                for expansion in orig_expansions[:3]:  # Limit expansions per keyword
                    if expansion != orig.lower():
                        keyword_pairs.append(f"{orig} {expansion}")

            # Add pairs to search queries
            for pair in keyword_pairs:
                if len(search_queries) >= max_combinations:
                    break
                search_queries.append(pair)

        logger.info(f"Generated {len(search_queries)} keyword combinations for search")
        return search_queries[:max_combinations]

    def score_dataset_relevance(
        self,
        dataset: Dict,
        keywords: List[str],
        boost_exact_match: float = 2.0
    ) -> float:
        """
        Score how relevant a dataset is to the given keywords.

        Args:
            dataset: Dataset metadata dict
            keywords: List of search keywords
            boost_exact_match: Multiplier for exact keyword matches

        Returns:
            Relevance score (higher = more relevant)
        """
        title = dataset.get("title", "").lower()
        description = dataset.get("desc", dataset.get("description", "")).lower()

        score = 0.0

        for keyword in keywords:
            keyword_lower = keyword.lower()

            # Exact match in title (highest weight)
            if keyword_lower in title:
                score += 10.0 * boost_exact_match

            # Exact match in description
            if keyword_lower in description:
                score += 5.0

            # Check expanded keywords
            expansions = self.expand_keyword(keyword)
            for expansion in expansions:
                if expansion in title:
                    score += 8.0  # Slightly lower than exact match
                if expansion in description:
                    score += 4.0

        # Boost for authorized publishers
        publisher = dataset.get("org", {}).get("title", "") if isinstance(dataset.get("org"), dict) else ""
        trusted_publishers = ["India Meteorological Department", "Ministry of Agriculture"]
        if any(trusted in publisher for trusted in trusted_publishers):
            score *= 1.5

        return score

    def rank_datasets(
        self,
        datasets: List[Dict],
        keywords: List[str]
    ) -> List[Tuple[Dict, float]]:
        """
        Rank datasets by relevance to keywords.

        Args:
            datasets: List of dataset metadata dicts
            keywords: Search keywords

        Returns:
            List of (dataset, score) tuples sorted by score (descending)
        """
        scored_datasets = [
            (dataset, self.score_dataset_relevance(dataset, keywords))
            for dataset in datasets
        ]

        # Sort by score descending
        scored_datasets.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Ranked {len(datasets)} datasets. Top score: {scored_datasets[0][1]:.2f}, Bottom: {scored_datasets[-1][1]:.2f}")

        return scored_datasets
