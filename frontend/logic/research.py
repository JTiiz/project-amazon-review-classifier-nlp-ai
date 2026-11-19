from __future__ import annotations

import logging
import os
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class ResearchGenerator:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        
        if not api_key or api_key.lower() == "none":
            self.llm = None
        else:
            self.llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                temperature=0.7,
                max_tokens=900,          # hard stop if the model starts looping
                frequency_penalty=0.3,   # discourages repeated tokens (e.g. runaway URLs)
            )
            
        logger.info("ResearchGenerator initialized | model=%s", model_name)

    def generate(self, product: str | None = None, category: str | None = None):
        query_term = product or category or "trending products"
        
        if not self.llm:
            yield "🛡️ **Operational Warning**: Intelligence protocol is not fully initialized. "
            yield "Please configure your OpenAI credentials in the system settings."
            return

        prompt = ChatPromptTemplate.from_template("""
        System: You are an expert Market Intelligence Analyst.
        Task: Synthesize a GROUNDED, revenue-focused intelligence report for "{query_term}".
        
        CRITICAL RULES:
        1. GROUNDED TRUTH: Provide only verified market insights, growth rates, and trends based on your deep knowledge base.
        2. Citations: Attribute every quantitative claim to a known industry source as a markdown link to the source's
           HOME PAGE only, e.g. [Statista](https://www.statista.com). Never write deep links or long URLs.
        3. ACCURACY: Use known, reliable industry sources (e.g., Gartner, Statista, Reuters, Bloomberg).
        
        Formatting:
        -catchy, vibrant title.
        -Revenue Snapshot: Key growth rates and market caps with source links.
        -Market Drivers: Why this category is trending with source links.
        -Grounded Data Sources: List the top 3-5 sources referenced above.
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            yield from chain.stream({"query_term": query_term})
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            yield f"Error generating research report: {str(e)}"
