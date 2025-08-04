# DSPy LangGraph Financial Pipeline

This repository implements a modular, dynamic financial analysis pipeline using [DSPy](https://github.com/stanfordnlp/dspy), [LangGraph](https://github.com/langchain-ai/langgraph), and web search/browsing tools. The system can answer complex finance-related queries by orchestrating multiple modules for stock data, fundamentals, and news analysis, using both LLM-based planning and real-time web data extraction.

---

## Features

- **Dynamic Query Planning:**  
  Uses a DSPy-powered planner to determine which modules (stock data, fundamentals, news analysis) are relevant for a given user query.

- **Web Search & Browsing:**  
  Integrates DuckDuckGo/DDGS search and Selenium-based browsing to extract up-to-date financial data and news from the web.

- **Modular DSPy Components:**  
  Each module (StockData, Fundamentals, NewsAnalysis) is implemented as a DSPy module with its own prompt and extraction logic.

- **LangGraph Workflow:**  
  Dynamically builds and executes a LangGraph workflow based on the planner's output, combining results and citations from each module.

- **Prompt & Module Optimization:**  
  Includes Jupyter notebooks for optimizing DSPy modules using few-shot learning and prompt search (see `optimise.ipynb`).

---

## File Overview

- **flow3.py**  
  Main pipeline implementation using DSPy, LangGraph, and Selenium. Handles query planning, module orchestration, and result aggregation.

- **base_flow.py**  
  Simpler, baseline version of the pipeline for comparison and ablation.

- **semi_aut_flow.py**  
  Semi-automated pipeline using DDGS for search and improved logging.

- **optimise.ipynb**  
  Jupyter notebook for optimizing DSPy modules and prompts using synthetic datasets.

- **stock_price_cot.json, portfolio_cot.json**  
  Saved optimized DSPy modules for stock price movement and portfolio recommendations.

---

## Requirements

- Python 3.10+
- [DSPy](https://github.com/stanfordnlp/dspy)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [langchain](https://github.com/langchain-ai/langchain)
- [selenium](https://pypi.org/project/selenium/)
- [ddgs](https://pypi.org/project/ddgs/) (DuckDuckGo Search)
- Chrome WebDriver (for Selenium)
- [dotenv](https://pypi.org/project/python-dotenv/) (for API keys)
- A supported LLM API key (e.g., Groq, OpenAI, etc.)

Install dependencies:
```sh
pip install dspy langgraph langchain selenium ddgs python-dotenv
```

---

## Usage

1. **Set up environment variables:**  
   Create a `.env` file with your LLM API key, e.g.:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

2. **Run the main pipeline:**  
   ```sh
   python flow3.py
   ```
   This will process example queries and print answers with citations.

3. **Optimize or evaluate modules:**  
   Open `optimise.ipynb` in Jupyter and run the cells to optimize DSPy modules or evaluate on synthetic datasets.

---

## Example Query

```
What is Apple's current P/E ratio?
```

**Output:**
```
Stock Data: $210.46 Fundamentals: P/E ratio: 32.71 News: Apple reported strong iPhone sales and record revenue.
Citations: [list of URLs]
```

---

## Customization

- **Add new modules:**  
  Implement a new DSPy module and update the planner prompt and workflow builder.
- **Change search backend:**  
  Swap out DDGS for another search API if needed.
- **Improve prompts:**  
  Tune prompts in each module for better extraction and summarization.

---

## Troubleshooting

- **Selenium errors:**  
  Ensure ChromeDriver is installed and matches your Chrome version.
- **API errors:**  
  Check your `.env` file and API key validity.
- **Search backend warnings:**  
  If you see warnings about `duckduckgo_search`, switch to `ddgs` as shown in the code.

---

## References

- [DSPy Documentation](https://stanfordnlp.github.io/dspy/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Tools](https://python.langchain.com/docs/integrations/tools/)
- [DDGS (DuckDuckGo Search)](https://pypi.org/project/ddgs/)

---

## License

MIT License (see repository for details).

---

**Contributions welcome!**  
Feel free to open issues or submit pull requests for improvements or new
