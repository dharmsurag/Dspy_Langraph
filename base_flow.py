import langgraph
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Annotated
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_core.tools import Tool
import operator
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import dspy
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure DSPy with Groq LLaMA model
lm = dspy.LM("groq/llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
dspy.configure(lm=lm)

# Define DSPy modules
class QueryAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(
            "query -> intents",
            prompt="""
            Classify the query into one or more of the following intents: stock_data, fundamentals, news.
            If the query does not match any of these intents, return an empty list.
            Return the intents as a JSON list.

            Examples:
            - Query: "What is Apple's current stock price?" Intents: ["stock_data"]
            - Query: "What is Apple's P/E ratio?" Intents: ["fundamentals"]
            - Query: "Is Apple a good investment?" Intents: ["stock_data", "fundamentals", "news"]
            - Query: "What's the weather today?" Intents: []

            Query: {query} Intents:
            """
        )

    def forward(self, query):
        result = self.predict(query=query)
        try:
            intents = json.loads(result.intents)
        except json.JSONDecodeError:
            intents = [result.intents] if result.intents else []
        return dspy.Prediction(intents=intents)

class StockDataModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.search = dspy.Predict(
            "query -> search_query",
            prompt="""
            Generate a search query to find stock prices or trends for the given query.
            Focus on real-time or historical stock data.

            Examples:
            - Query: "What is Apple's current stock price?" Search Query: "Apple stock price"
            - Query: "Apple stock trend last month" Search Query: "Apple stock price history last month"

            Query: {query} Search Query:
            """
        )
        self.extract = dspy.Predict(
            "contents, query -> stock_data",
            prompt="""
            Extract stock prices or trends from multiple webpage contents based on the query.
            If numerical data (e.g., prices), compute the average if consistent, or select the most recent.
            If trends, summarize the direction (e.g., rising, falling).
            Return the aggregated result.

            Examples:
            - Query: "What is Apple's current stock price?" Contents: ["...AAPL $210.42...", "...AAPL $210.50..."] Stock Data: "$210.46"
            - Query: "Apple stock trend last month" Contents: ["...AAPL rose 5%...", "...AAPL up 4.8%..."] Stock Data: "Rose approximately 4.9% last month"

            Contents: {contents} Query: {query} Stock Data:
            """
        )

    def forward(self, query):
        search_query = self.search(query=query).search_query
        search_results = text_search_tool.run(search_query)
        urls = [result['link'] for result in search_results if 'link' in result][:3]  # Take top 3 URLs
        if not urls:
            return {"stock_data": "No stock data found", "sources": []}
        contents = []
        sources = []
        for url in urls:
            content = browse_tool.run({"url": url, "query": query})
            if not content.startswith("Error"):
                contents.append(content)
                sources.append(url)
        if not contents:
            return {"stock_data": "No stock data found", "sources": []}
        result = self.extract(contents=contents, query=query)
        return {"stock_data": result.stock_data, "sources": sources}

class FundamentalsModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.search = dspy.Predict(
            "query -> search_query",
            prompt="""
            Generate a search query to find financial metrics or statements for the given query.
            Focus on ratios (e.g., P/E, EPS) or financial statements.

            Examples:
            - Query: "What is Apple's P/E ratio?" Search Query: "Apple P/E ratio"
            - Query: "Apple revenue last quarter" Search Query: "Apple financial statements revenue"

            Query: {query} Search Query:
            """
        )
        self.extract = dspy.Predict(
            "contents, query -> fundamentals_data",
            prompt="""
            Extract financial metrics or statements from multiple webpage contents based on the query.
            For numerical metrics (e.g., P/E ratio), select the most frequent or average if consistent.
            For statements, summarize key points.
            Return the aggregated result.

            Examples:
            - Query: "What is Apple's P/E ratio?" Contents: ["...P/E Ratio: 32.72...", "...P/E: 32.70..."] Fundamentals Data: "P/E ratio: 32.71"
            - Query: "Apple revenue last quarter" Contents: ["...Revenue: $85.8 billion...", "...Revenue: $85.7 billion..."] Fundamentals Data: "Revenue: $85.75 billion"

            Contents: {contents} Query: {query} Fundamentals Data:
            """
        )

    def forward(self, query):
        search_query = self.search(query=query).search_query
        search_results = text_search_tool.run(search_query)
        urls = [result['link'] for result in search_results if 'link' in result][:3]  # Take top 3 URLs
        if not urls:
            return {"fundamentals_data": "No fundamentals data found", "sources": []}
        contents = []
        sources = []
        for url in urls:
            content = browse_tool.run({"url": url, "query": query})
            if not content.startswith("Error"):
                contents.append(content)
                sources.append(url)
        if not contents:
            return {"fundamentals_data": "No fundamentals data found", "sources": []}
        result = self.extract(contents=contents, query=query)
        return {"fundamentals_data": result.fundamentals_data, "sources": sources}

class NewsAnalysisModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.search = dspy.Predict(
            "query -> search_query",
            prompt="""
            Generate a search query to find news articles or analyst opinions for the given query.
            Focus on recent company news or earnings reports.

            Examples:
            - Query: "Apple latest earnings news" Search Query: "Apple earnings news"
            - Query: "Is Apple a good investment?" Search Query: "Apple investment analysis news"

            Query: {query} Search Query:
            """
        )
        self.extract = dspy.Predict(
            "contents, query -> news_summary",
            prompt="""
            Summarize news articles or analyst opinions from multiple webpage contents based on the query.
            Combine key points into a concise summary, prioritizing consistency across sources.

            Examples:
            - Query: "Apple latest earnings news" Contents: ["...strong iPhone sales...", "...record iPhone revenue..."] News Summary: "Apple reported strong iPhone sales and record revenue."
            - Query: "Is Apple a good investment?" Contents: ["...rated Buy by analysts...", "...strong growth outlook..."] News Summary: "Analysts rate Apple as a Buy, citing strong growth."

            Contents: {contents} Query: {query} News Summary:
            """
        )

    def forward(self, query):
        search_query = self.search(query=query).search_query
        search_results = news_search_tool.run(search_query)
        urls = [result['link'] for result in search_results if 'link' in result][:3]  # Take top 3 URLs
        if not urls:
            return {"news_summary": "No news found", "sources": []}
        contents = []
        sources = []
        for url in urls:
            content = browse_tool.run({"url": url, "query": query})
            if not content.startswith("Error"):
                contents.append(content)
                sources.append(url)
        if not contents:
            return {"news_summary": "No news found", "sources": []}
        result = self.extract(contents=contents, query=query)
        return {"news_summary": result.news_summary, "sources": sources}

# Define LangGraph state
class FinanceState(TypedDict):
    query: Annotated[str, operator.add]
    intents: List[str]
    results: dict
    answer: str
    citations: List[str]

# Initialize tools
news_search_tool = DuckDuckGoSearchResults(backend="news", max_results=5, verbose=True, output_format="list")
text_search_tool = DuckDuckGoSearchResults(backend="text", max_results=5, verbose=True, output_format="list")

n_search_tool = Tool(
    name="News Search",
    func=news_search_tool.invoke,
    description="Search the web for recent news articles."
)

t_search_tool = Tool(
    name="Text Search",
    func=text_search_tool.invoke,
    description="Search the web for financial data."
)

# Selenium-based browse tool
def browse_page_selenium(input_dict: dict) -> str:
    url = input_dict["url"]
    query = input_dict["query"]
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get(url)
        content = driver.find_element("tag name", "body").text
        return content
    except Exception as e:
        return f"Error browsing page: {str(e)}"
    finally:
        driver.quit()

browse_tool = Tool(
    name="Browse Page",
    func=browse_page_selenium,
    description="Navigate to a URL and extract specific financial data or news."
)

# Define LangGraph nodes
def query_analyzer_node(state: FinanceState) -> FinanceState:
    analyzer = QueryAnalyzer()
    result = analyzer(state["query"])
    state["intents"] = result.intents
    return state

def stock_data_node(state: FinanceState) -> FinanceState:
    if "stock_data" in state["intents"]:
        module = StockDataModule()
        result = module(state["query"])
        state["results"]["stock_data"] = result["stock_data"]
        state["citations"].extend(result["sources"])
    return state

def fundamentals_node(state: FinanceState) -> FinanceState:
    if "fundamentals" in state["intents"]:
        module = FundamentalsModule()
        result = module(state["query"])
        state["results"]["fundamentals_data"] = result["fundamentals_data"]
        state["citations"].extend(result["sources"])
    return state

def news_analysis_node(state: FinanceState) -> FinanceState:
    if "news" in state["intents"]:
        module = NewsAnalysisModule()
        result = module(state["query"])
        state["results"]["news_summary"] = result["news_summary"]
        state["citations"].extend(result["sources"])
    return state

def combine_results_node(state: FinanceState) -> FinanceState:
    results = state["results"]
    answer_parts = []
    if "stock_data" in results:
        answer_parts.append(f"Stock Data: {results['stock_data']}")
    if "fundamentals_data" in results:
        answer_parts.append(f"Fundamentals: {results['fundamentals_data']}")
    if "news_summary" in results:
        answer_parts.append(f"News: {results['news_summary']}")
    answer = " ".join(answer_parts) if answer_parts else "No relevant data found for the query."
    # Remove duplicate citations while preserving order
    seen = set()
    citations = [c for c in state["citations"] if not (c in seen or seen.add(c))]
    return {"answer": answer, "citations": citations}

# Build the LangGraph workflow
workflow = StateGraph(FinanceState)

# Add nodes
workflow.add_node("query_analyzer", query_analyzer_node)
workflow.add_node("stock_data", stock_data_node)
workflow.add_node("fundamentals", fundamentals_node)
workflow.add_node("news_analysis", news_analysis_node)
workflow.add_node("combine_results", combine_results_node)

# Define edges
workflow.add_edge("query_analyzer", "stock_data")
workflow.add_edge("query_analyzer", "fundamentals")
workflow.add_edge("query_analyzer", "news_analysis")
workflow.add_edge("stock_data", "combine_results")
workflow.add_edge("fundamentals", "combine_results")
workflow.add_edge("news_analysis", "combine_results")
workflow.set_entry_point("query_analyzer")
workflow.set_finish_point("combine_results")

# Compile the graph
graph = workflow.compile()

# Example usage
def run_pipeline(query: str) -> dict:
    state = FinanceState(
        query=query,
        intents=[],
        results={},
        answer="",
        citations=[]
    )
    result = graph.invoke(state)
    return {"answer": result["answer"], "citations": result["citations"]}

# Run the pipeline
if __name__ == "__main__":
    
    # Example queries
    queries = [
        "What is Apple's current P/E ratio?",
        "Is Apple a good investment?",
        "What's the weather today?"
    ]
    for query in queries:
        result = run_pipeline(query)
        print(f"Query: {query}")
        print(f"Answer: {result['answer']}")
        print(f"Citations: {result['citations']}\n")