###DSPy LangGraph Financial Pipeline
This repository implements a modular, dynamic financial analysis pipeline using DSPy, LangGraph, and web search/browsing tools. The system can answer complex finance-related queries by orchestrating multiple modules for stock data, fundamentals, and news analysis, using both LLM-based planning and real-time web data extraction.

Features
Dynamic Query Planning:
Uses a DSPy-powered planner to determine which modules (stock data, fundamentals, news analysis) are relevant for a given user query.

Web Search & Browsing:
Integrates DuckDuckGo/DDGS search and Selenium-based browsing to extract up-to-date financial data and news from the web.

Modular DSPy Components:
Each module (StockData, Fundamentals, NewsAnalysis) is implemented as a DSPy module with its own prompt and extraction logic.

LangGraph Workflow:
Dynamically builds and executes a LangGraph workflow based on the planner's output, combining results and citations from each module.

Prompt & Module Optimization:
Includes Jupyter notebooks for optimizing DSPy modules using few-shot learning and prompt search (see optimise.ipynb).
