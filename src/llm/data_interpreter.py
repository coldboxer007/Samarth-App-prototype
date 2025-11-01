"""LLM-based data interpreter that works directly with raw datasets."""

import pandas as pd
from typing import List, Dict, Any, Optional
import logging
import google.generativeai as genai
import json
import os

from src.config import GEMINI_API_KEY

# Disable Google Cloud default credentials to prevent metadata service errors on Streamlit Cloud
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''
os.environ['NO_GCE_CHECK'] = 'True'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataInterpreter:
    """Interprets raw datasets using LLM to answer questions directly."""

    # State name mappings for historical variations
    STATE_NAME_MAPPINGS = {
        'odisha': ['orissa', 'odisha'],
        'mumbai': ['bombay', 'mumbai'],
        'chennai': ['madras', 'chennai'],
        'kolkata': ['calcutta', 'kolkata'],
        'bengaluru': ['bangalore', 'bengaluru'],
        'uttarakhand': ['uttaranchal', 'uttarakhand'],
        'chhattisgarh': ['chattisgarh', 'chhattisgarh'],
        'telangana': ['telengana', 'telangana'],
    }

    def __init__(self, api_key: str = GEMINI_API_KEY):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required. Please add it to Streamlit secrets or .env file")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')

    def filter_dataset_for_query(
        self,
        user_question: str,
        df: pd.DataFrame,
        dataset_name: str
    ) -> pd.DataFrame:
        """
        Use LLM to intelligently filter a large dataset based on the query.

        Args:
            user_question: User's question
            df: Full DataFrame
            dataset_name: Name of dataset for context

        Returns:
            Filtered DataFrame with only relevant rows
        """
        if len(df) <= 100:
            # Small dataset, no need to filter
            return df

        logger.info(f"Filtering {dataset_name} ({len(df)} rows) for query relevance")

        # Get column info and sample
        columns_info = ", ".join([f"{col} ({df[col].dtype})" for col in df.columns])
        sample = df.head(5).to_string(index=False)

        # Extract state name mappings that might be relevant to the question
        state_mappings_hint = ""
        question_lower = user_question.lower()
        for current_name, variants in self.STATE_NAME_MAPPINGS.items():
            if current_name in question_lower or any(v in question_lower for v in variants):
                state_mappings_hint += f"\n- '{current_name.title()}' may be stored as: {', '.join([v.upper() for v in variants])}"

        filter_prompt = f"""You are a data filtering assistant. Given a user's question and a dataset, determine which rows are relevant.

**User Question:** {user_question}

**Dataset:** {dataset_name}
**Total Rows:** {len(df)}
**Columns:** {columns_info}

**Sample Data:**
```
{sample}
```

**IMPORTANT - State Name Variations:**
Historical datasets may use old names for states/cities. Always check for variations:{state_mappings_hint}

**Task:** Write a Python pandas filter expression to keep only relevant rows.

**Examples:**
- Question: "rainfall in Maharashtra" → `df[df['state'].str.contains('Maharashtra', case=False, na=False)]`
- Question: "rainfall in Odisha" → `df[df['state'].str.contains('Odisha|Orissa', case=False, na=False)]` (use both variants!)
- Question: "rice production last 5 years" → `df[(df['crop'] == 'Rice') & (df['year'] >= 2019)]`
- Question: "top districts" → `df` (no filter, need all for ranking)

**Rules:**
1. Return ONLY the filter expression, nothing else
2. Use df as the variable name
3. Handle missing values with na=False
4. Use case-insensitive matching with case=False
5. For state/city names, use regex OR pattern to include historical variants: 'Odisha|Orissa'
6. If no filtering needed (e.g., need all data for aggregation), return: `df`

Filter expression:"""

        try:
            response = self.model.generate_content(filter_prompt)
            filter_expr = response.text.strip()

            # Remove markdown code blocks if present
            if filter_expr.startswith("```"):
                filter_expr = filter_expr.split("```")[1]
                if filter_expr.startswith("python"):
                    filter_expr = filter_expr[6:]
                filter_expr = filter_expr.strip()

            # Clean up the expression
            filter_expr = filter_expr.replace("```", "").strip()

            logger.info(f"LLM filter expression: {filter_expr}")

            # If LLM says no filter needed, return original
            if filter_expr == "df" or "# no filter" in filter_expr.lower():
                logger.info("No filtering needed per LLM")
                return df

            # Execute the filter
            filtered_df = eval(filter_expr, {"df": df, "pd": pd})

            logger.info(f"Filtered from {len(df)} to {len(filtered_df)} rows")

            return filtered_df

        except Exception as e:
            logger.warning(f"Error filtering dataset: {e}, using original")
            return df

    def interpret_and_answer(
        self,
        user_question: str,
        datasets: List[Dict[str, Any]],
        max_rows_per_dataset: int = 1000,
        auto_filter: bool = True
    ) -> Dict[str, Any]:
        """
        Answer user question by interpreting raw datasets directly.

        Args:
            user_question: The user's question
            datasets: List of dataset dicts with 'name', 'data' (DataFrame), 'metadata'
            max_rows_per_dataset: Maximum rows to send to LLM per dataset

        Returns:
            Dict with 'answer' (narrative), 'data_used' (summary), 'sources'
        """
        logger.info(f"Interpreting {len(datasets)} datasets to answer: {user_question}")

        # Prepare datasets for LLM consumption with optional filtering
        dataset_summaries = []
        for ds in datasets:
            df = ds.get('data')
            name = ds.get('name', 'Unknown')

            # Apply intelligent filtering for large datasets
            if auto_filter and df is not None and len(df) > 100:
                df = self.filter_dataset_for_query(user_question, df, name)

            summary = self._prepare_dataset_summary(
                name=name,
                df=df,
                metadata=ds.get('metadata', {}),
                max_rows=max_rows_per_dataset
            )
            dataset_summaries.append(summary)

        # Build prompt for LLM
        prompt = self._build_interpretation_prompt(
            question=user_question,
            dataset_summaries=dataset_summaries
        )

        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()

            logger.info("LLM successfully interpreted datasets")

            return {
                "answer": result_text,
                "datasets_used": [ds.get('name', 'Unknown') for ds in datasets],
                "sources": [ds.get('name', 'Unknown') for ds in datasets]
            }

        except Exception as e:
            logger.error(f"Error interpreting datasets: {e}")
            raise

    def _prepare_dataset_summary(
        self,
        name: str,
        df: pd.DataFrame,
        metadata: Dict,
        max_rows: int = 1000
    ) -> str:
        """
        Convert dataset to LLM-friendly summary format.

        Format optimized for LLM interpretation:
        - Dataset name and metadata
        - Column names and types
        - Sample rows (smart sampling)
        - Summary statistics for numeric columns
        """
        if df is None or df.empty:
            return f"**Dataset: {name}**\nNo data available.\n"

        summary_parts = []

        # Header
        summary_parts.append(f"**Dataset: {name}**")
        summary_parts.append(f"Total rows: {len(df)}, Columns: {len(df.columns)}")

        # Metadata
        if metadata:
            summary_parts.append(f"Source ID: {metadata.get('source_id', 'unknown')}")
            if 'description' in metadata:
                summary_parts.append(f"Description: {metadata['description']}")

        # Column information
        summary_parts.append("\n**Columns:**")
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            summary_parts.append(f"  - {col} ({dtype}) - {non_null}/{len(df)} non-null")

        # Smart sampling: First few, last few, and middle if dataset is large
        sample_df = self._smart_sample(df, max_rows)

        # Add sample data in a clean table format
        summary_parts.append("\n**Sample Data:**")
        summary_parts.append("```")
        summary_parts.append(sample_df.to_string(index=False, max_rows=50))
        summary_parts.append("```")

        # Summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols and len(numeric_cols) > 0:
            summary_parts.append("\n**Numeric Summary Statistics:**")
            summary_parts.append("```")
            summary_parts.append(df[numeric_cols].describe().to_string())
            summary_parts.append("```")

        return "\n".join(summary_parts)

    def _smart_sample(self, df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        """
        Smart sampling strategy to give LLM representative view of data.

        Strategy:
        - If df <= max_rows: return all
        - Else: return first 40%, middle 20%, last 40%
        """
        if len(df) <= max_rows:
            return df

        # Calculate split points
        first_n = int(max_rows * 0.4)
        middle_n = int(max_rows * 0.2)
        last_n = max_rows - first_n - middle_n

        middle_start = (len(df) - middle_n) // 2

        # Combine samples
        samples = pd.concat([
            df.head(first_n),
            df.iloc[middle_start:middle_start + middle_n],
            df.tail(last_n)
        ])

        return samples

    def _build_interpretation_prompt(
        self,
        question: str,
        dataset_summaries: List[str]
    ) -> str:
        """Build prompt for LLM to interpret datasets and answer question."""

        system_prompt = """You are an experienced professor of agricultural and climate sciences with decades of expertise in Indian agriculture, meteorology, and regional patterns. You're having a conversation with a student or colleague, sharing insights in a warm, educational, and engaging manner.

IMPORTANT - STATE NAME VARIATIONS:
Historical datasets often use old names. Always check for these variations:
- Odisha → may be stored as "ORISSA" (pre-2011 name)
- Uttarakhand → may be "UTTARANCHAL"
- Chhattisgarh → may be "CHATTISGARH"
- Mumbai → may be "BOMBAY"
- Chennai → may be "MADRAS"
- Kolkata → may be "CALCUTTA"
- Bengaluru → may be "BANGALORE"
- Telangana → may be "TELENGANA"

YOUR COMMUNICATION STYLE:
1. **Conversational and warm** - Write as if explaining to a student over coffee
2. **Start with the answer** - Give the specific numbers/facts first, then explain
3. **Tell a story with the data** - Help your audience understand WHY these numbers matter
4. **Use transitions naturally** - "What's interesting here is...", "Let me explain...", "Here's what this means..."
5. **Share your expertise** - Use phrases like "In my experience...", "What we typically see is...", "This reminds me of..."
6. **Be encouraging and insightful** - Not just data, but wisdom

DATA INTERPRETATION GUIDELINES:
- **Year ranges**: "2003-04" means year 2003 (use first year)
- **State names**: ALWAYS check for historical variants (Odisha/Orissa, etc.)
- **Indian units**: 1 lakh = 100,000; 1 crore = 10,000,000
- **Agricultural seasons**: Kharif (June-Oct), Rabi (Oct-March), Zaid (March-June)
- **Regional patterns**: Use your deep knowledge of India's climate zones, soil types, cropping patterns

WHAT TO INCLUDE IN YOUR RESPONSE:
- **The answer** (with specific numbers from data)
- **Context** (what's typical, how this compares)
- **Why it matters** (implications for agriculture, livelihoods)
- **Insights** (patterns, trends, interesting observations)
- **Practical wisdom** (what farmers/planners should consider)

OUTPUT FORMAT (CONVERSATIONAL & EDUCATIONAL):

[Start with a natural greeting to the answer - "Looking at the data, we can see that..." or "Let me share what I found..."]

[Give the specific answer with numbers - but in a conversational way, not bullet points]

[Explain the context and what makes this interesting or significant - relate to broader patterns]

[Share implications and practical wisdom - what this means in the real world]

[If needed, suggest next steps or additional considerations]

Data sources: [List naturally, like: "This analysis is based on..."]

EXAMPLES:

❌ BAD (robotic, just facts):
"The average rainfall is 850mm based on data from 2019-2024.

Data sources: Rainfall Data"

❌ BAD (too formal, sounds like a report):
"**Analysis:** Based on the provided dataset...
**Findings:** The analysis reveals...
**Conclusion:** The data indicates..."

✅ EXCELLENT (conversational and educational):
"Looking at the data for Maharashtra from 2019 to 2024, we can see the average annual rainfall is around 850mm. Now, what's interesting here is that this is actually about 15% below the state's historical average of around 1000mm.

Why does this matter? Well, Maharashtra is one of India's key agricultural states, and this rainfall deficit directly impacts their major crops. Take sugarcane and cotton, for instance - these crops typically need about 1000-1200mm of rainfall annually to thrive. With the current patterns, farmers are facing a real challenge.

In my experience working with agricultural data across India, what we're seeing here is part of a broader trend of changing monsoon patterns, particularly affecting the Western Ghats region. For farmers in Maharashtra, this means they might want to consider shifting towards more drought-resistant crop varieties or investing in micro-irrigation systems. It's not just about adapting to one bad year - it's about planning for a changing climate.

This analysis is based on rainfall data from the India Meteorological Department covering the 2019-2024 period."

✅ EXCELLENT (handling missing data gracefully):
"I looked through the available datasets for Odisha's rainfall in 1951, and here's the situation - the data for that specific year is recorded under the old name 'ORISSA'. Let me tell you what we find: the average annual rainfall for Orissa in 1951 was 1396.3mm.

This is actually quite typical for Odisha. The state usually receives between 1400-1500mm annually, and 1951 falls right in that range. What makes Odisha's rainfall pattern interesting is the strong influence of the Bay of Bengal - the coastal districts typically see higher amounts, sometimes reaching 1500-1800mm, while the interior regions are a bit drier.

If you're looking at this from an agricultural planning perspective, this 1951 data point is valuable because it represents a fairly normal monsoon year. Odisha's agriculture has traditionally been built around expecting this level of rainfall, which supports their main crops like rice, which thrives in these conditions.

The data comes from the Area Weighted Monthly, Seasonal and Annual Rainfall records for Indian meteorological subdivisions, which uses the historical name 'ORISSA' for this state."

Now, share your insights in this warm, conversational, educational style:
"""

        # Combine all dataset summaries
        datasets_section = "\n\n" + "="*80 + "\n".join(dataset_summaries) + "\n" + "="*80

        user_section = f"""

**USER QUESTION:**
{question}

**AVAILABLE DATASETS:**
{datasets_section}

Please analyze these datasets and answer the question above.
"""

        return system_prompt + user_section

    def extract_structured_data(
        self,
        user_question: str,
        df: pd.DataFrame,
        dataset_name: str
    ) -> pd.DataFrame:
        """
        Ask LLM to extract and structure specific data from a raw dataset.

        This is useful when you need structured output for further processing.

        Args:
            user_question: What data to extract (e.g., "Extract state, year, and rainfall")
            df: Raw DataFrame
            dataset_name: Name for context

        Returns:
            Structured DataFrame based on LLM interpretation
        """
        logger.info(f"Extracting structured data from {dataset_name}")

        # Prepare dataset summary
        summary = self._prepare_dataset_summary(
            name=dataset_name,
            df=df,
            metadata={},
            max_rows=100  # Use smaller sample for extraction
        )

        prompt = f"""You are a data extraction assistant. Extract and structure data from the raw dataset below.

**TASK:**
{user_question}

**RAW DATASET:**
{summary}

**OUTPUT FORMAT:**
Return a JSON array of objects, where each object represents one row of extracted data.

Example format:
[
    {{"state": "Maharashtra", "year": 2020, "rainfall_mm": 850.5}},
    {{"state": "Punjab", "year": 2020, "rainfall_mm": 650.2}},
    ...
]

IMPORTANT:
- Extract ALL relevant rows from the dataset (not just samples)
- Handle data transformations (e.g., "2003-04" → 2003)
- Convert units as needed
- Use clear, descriptive field names
- Return ONLY valid JSON, no additional text

Now extract the data:
"""

        try:
            response = self.model.generate_content(prompt)
            json_text = response.text.strip()

            # Remove markdown code blocks if present
            if json_text.startswith("```"):
                json_text = json_text.split("```")[1]
                if json_text.startswith("json"):
                    json_text = json_text[4:]
                json_text = json_text.strip()

            # Parse JSON
            extracted_data = json.loads(json_text)

            # Convert to DataFrame
            result_df = pd.DataFrame(extracted_data)

            logger.info(f"Extracted {len(result_df)} structured records")

            return result_df

        except Exception as e:
            logger.error(f"Error extracting structured data: {e}")
            raise
