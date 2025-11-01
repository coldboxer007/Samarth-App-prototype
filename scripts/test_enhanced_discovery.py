"""Test script for enhanced dataset discovery with keyword expansion and parallel search."""

import sys
sys.path.append('.')

from src.catalog.dataset_discovery import DatasetDiscovery
from src.catalog.keyword_expander import KeywordExpander
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_keyword_expansion():
    """Test keyword expansion functionality."""
    print("\n" + "="*80)
    print("TEST 1: Keyword Expansion")
    print("="*80)

    expander = KeywordExpander()

    # Test single keyword expansion
    test_keywords = ["rainfall", "odisha", "crop", "agriculture"]

    for keyword in test_keywords:
        expanded = expander.expand_keyword(keyword)
        print(f"\nKeyword: '{keyword}'")
        print(f"Expanded to ({len(expanded)} variations): {expanded[:5]}...")

    # Test keyword combinations
    print("\n" + "-"*80)
    print("Generating Keyword Combinations")
    print("-"*80)

    user_keywords = ["rainfall", "odisha"]
    combinations = expander.generate_keyword_combinations(
        keywords=user_keywords,
        max_combinations=15,
        include_single=True,
        include_pairs=True,
        expand_keywords=True
    )

    print(f"\nOriginal keywords: {user_keywords}")
    print(f"Generated {len(combinations)} search combinations:")
    for i, combo in enumerate(combinations, 1):
        print(f"  {i}. {combo}")


def test_parallel_search():
    """Test parallel dataset search."""
    print("\n" + "="*80)
    print("TEST 2: Parallel Dataset Search")
    print("="*80)

    discovery = DatasetDiscovery()

    # Create multiple search queries
    search_queries = [
        "rainfall",
        "precipitation",
        "monsoon",
        "odisha rainfall",
        "orissa rainfall"
    ]

    print(f"\nSearching with {len(search_queries)} queries in parallel:")
    for i, query in enumerate(search_queries, 1):
        print(f"  {i}. {query}")

    # Perform parallel search
    import time
    start_time = time.time()

    datasets = discovery.search_datasets_parallel(
        queries=search_queries,
        max_results_per_query=5,
        max_workers=3
    )

    elapsed = time.time() - start_time

    print(f"\nParallel search completed in {elapsed:.2f} seconds")
    print(f"Found {len(datasets)} unique datasets")

    if datasets:
        print("\nSample dataset titles:")
        for i, ds in enumerate(datasets[:5], 1):
            title = ds.get('title', 'Unknown')
            print(f"  {i}. {title}")


def test_relevance_scoring():
    """Test dataset relevance scoring."""
    print("\n" + "="*80)
    print("TEST 3: Relevance Scoring")
    print("="*80)

    expander = KeywordExpander()

    # Mock datasets
    test_datasets = [
        {
            "title": "Rainfall Data for Odisha",
            "desc": "Monthly rainfall measurements from IMD for Odisha state",
            "org": {"title": "India Meteorological Department"}
        },
        {
            "title": "Crop Production Statistics",
            "desc": "Agricultural production data for various crops",
            "org": {"title": "Ministry of Agriculture"}
        },
        {
            "title": "Temperature Records",
            "desc": "Historical temperature data for Indian states",
            "org": {"title": "Weather Department"}
        },
        {
            "title": "Odisha Agricultural Statistics",
            "desc": "Crop yield and production data for Odisha",
            "org": {"title": "Ministry of Agriculture"}
        }
    ]

    # Test keywords
    keywords = ["rainfall", "odisha"]

    print(f"\nKeywords: {keywords}")
    print("\nDataset Relevance Scores:")

    # Score and rank datasets
    ranked = expander.rank_datasets(test_datasets, keywords)

    for i, (dataset, score) in enumerate(ranked, 1):
        title = dataset.get('title', 'Unknown')
        print(f"  {i}. [{score:.2f}] {title}")


def test_enhanced_question_discovery():
    """Test the enhanced discover_and_add_datasets_for_question method."""
    print("\n" + "="*80)
    print("TEST 4: Enhanced Question-Based Discovery")
    print("="*80)

    discovery = DatasetDiscovery()

    # Test question
    question = "What was the rainfall in Odisha in 1951?"

    print(f"\nQuestion: {question}")
    print("\nDiscovering datasets with enhanced search...")

    import time
    start_time = time.time()

    # Discover datasets with keyword expansion and parallel search
    dataset_ids = discovery.discover_and_add_datasets_for_question(
        question=question,
        max_combinations=15,
        max_workers=3
    )

    elapsed = time.time() - start_time

    print(f"\nDiscovery completed in {elapsed:.2f} seconds")
    print(f"Found {len(dataset_ids)} relevant datasets")

    if dataset_ids:
        print("\nRelevant Dataset IDs:")
        for i, ds_id in enumerate(dataset_ids[:10], 1):
            # Get dataset info from catalog
            dataset = discovery.catalog.get_dataset(ds_id)
            if dataset:
                print(f"  {i}. {ds_id}")
                print(f"     Name: {dataset.get('name', 'Unknown')}")
                print(f"     Category: {dataset.get('category', 'Unknown')}")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("ENHANCED DATASET DISCOVERY - COMPREHENSIVE TEST SUITE")
    print("="*80)

    try:
        # Test 1: Keyword expansion
        test_keyword_expansion()

        # Test 2: Parallel search
        test_parallel_search()

        # Test 3: Relevance scoring
        test_relevance_scoring()

        # Test 4: Full enhanced discovery
        test_enhanced_question_discovery()

        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
