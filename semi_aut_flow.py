import langgraph
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Annotated
from langchain_core.tools import Tool, StructuredTool
import operator
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import dspy
import json
import os
from dotenv import load_dotenv
import logging
from ddgs import DDGS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure DSPy with Groq LLaMA model
try:
    lm = dspy.LM("groq/llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    dspy.configure(lm=lm)
except Exception as e:
    logger.error(f"Failed to configure DSPy: {str(e)}")
    raise

# Define DSPy modules
class Planner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(
            "query -> plan",
            prompt="""
            Given the user's query, determine the sequence of modules to call.
            Available modules: stock_data, fundamentals, news_analysis
            Return the sequence as a JSON list. If the query is unrelated, return an empty list.

            Examples:
            - Query: "What is Apple's current stock price?" Plan: ["stock_data"]
            - Query: "What is Apple's P/E ratio?" Plan: ["fundamentals"]
            - Query: "Is Apple a good investment?" Plan: ["stock_data", "fundamentals", "news_analysis"]
            - Query: "What's the weather today?" Plan: []

            Query: {query} Plan:
            """
        )

    def forward(self, query):
        
        plan = []
        result = self.predict(query=query)
        print(f"Planner result: {result.plan}")
        # Ideally with a more sophisticated model, we would parse the plan directly.
        # For simplicity, we will use keyword matching to determine the plan.
        stock_keywords = ["stock", "stock_data", "price", "trend"]
        fundamentals_keywords = ["fundamentals", "ratio", "financial", "metrics", "statement", "P/E", "EPS"]
        news_keywords = ["news", "earnings", "analyst", "opinion", "report"]
        if any(keyword.lower() in result.plan.lower() for keyword in stock_keywords):
            plan.append("stock_data")
        if any(keyword.lower() in result.plan.lower() for keyword in fundamentals_keywords):
            plan.append("fundamentals")
        if any(keyword.lower() in result.plan.lower() for keyword in news_keywords):
            plan.append("news_analysis")
        print(f"Generated plan: {plan}")
        return plan

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

    def forward(self, state):
        query = state["query"]
        logger.info(f"StockDataModule received query: {query}")
        try:
            search_query = self.search(query=query).search_query
            logger.info(f"Search query: {search_query}")
            search_results = text_search_tool.run(search_query)
            logger.info(f"Search results: {search_results}")
            urls = [result['href'] for result in search_results if 'href' in result][:3]
            if not urls:
                logger.warning("No URLs found")
                return {"stock_data": "No stock data found", "sources": []}
            contents = []
            sources = []
            for url in urls:
                content = browse_tool.run({"url": url, "query": query})
                logger.info(f"Content from {url}: {content[:200]}...")
                if not content.startswith("Error"):
                    contents.append(content)
                    sources.append(url)
            if not contents:
                logger.warning("No valid contents found")
                return {"stock_data": "No stock data found", "sources": []}
            result = self.extract(contents="\n".join(contents), query=query)
            logger.info(f"Extracted result: {result}")
            return {"stock_data": result.stock_data, "sources": sources}
        except Exception as e:
            logger.error(f"Error in StockDataModule: {str(e)}")
            return {"stock_data": "Error processing stock data", "sources": []}

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

    def forward(self, state):
        query = state["query"]
        logger.info(f"FundamentalsModule received query: {query}")
        try:
            search_query = self.search(query=query).search_query
            logger.info(f"Search query: {search_query}")
            search_results = text_search_tool.run(search_query)
            logger.info(f"Search results: {search_results}")
            urls = [result['href'] for result in search_results if 'href' in result][:3]
            if not urls:
                logger.warning("No URLs found")
                return {"fundamentals_data": "No fundamentals data found", "sources": []}
            contents = []
            sources = []
            for url in urls:
                content = browse_tool.run({"url": url, "query": query})
                logger.info(f"Content from {url}: {content[:200]}...")
                if not content.startswith("Error"):
                    contents.append(content)
                    sources.append(url)
            if not contents:
                logger.warning("No valid contents found")
                return {"fundamentals_data": "No fundamentals data found", "sources": []}
            result = self.extract(contents="\n".join(contents), query=query)
            logger.info(f"Extracted result: {result}")
            return {"fundamentals_data": result.fundamentals_data, "sources": sources}
        except Exception as e:
            logger.error(f"Error in FundamentalsModule: {str(e)}")
            return {"fundamentals_data": "Error processing fundamentals data", "sources": []}

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

    def forward(self, state):
        query = state["query"]
        logger.info(f"NewsAnalysisModule received query: {query}")
        try:
            search_query = self.search(query=query).search_query
            logger.info(f"Search query: {search_query}")
            search_results = news_search_tool.run(search_query)
            logger.info(f"Search results: {search_results}")
            urls = [result['url'] for result in search_results if 'url' in result][:3]
            if not urls:
                logger.warning("No URLs found")
                return {"news_summary": "No news found", "sources": []}
            contents = []
            sources = []
            for url in urls:
                content = browse_tool.run({"url": url, "query": query})
                logger.info(f"Content from {url}: {content[:200]}...")
                if not content.startswith("Error"):
                    contents.append(content)
                    sources.append(url)
            if not contents:
                logger.warning("No valid contents found")
                return {"news_summary": "No news found", "sources": []}
            result = self.extract(contents="\n".join(contents), query=query)
            logger.info(f"Extracted result: {result}")
            return {"news_summary": result.news_summary, "sources": sources}
        except Exception as e:
            logger.error(f"Error in NewsAnalysisModule: {str(e)}")
            return {"news_summary": "Error processing news data", "sources": []}

# Define LangGraph state
class FinanceState(TypedDict):
    query: Annotated[str, operator.add]
    plan: List[str]
    results: dict
    answer: str
    citations: List[str]

# Initialize tools
def ddgs_search(query: str) -> list:
    try:
        with DDGS() as ddgs:
            return ddgs.text(query, max_results=5)
    except Exception as e:
        logger.error(f"Error in text search: {str(e)}")
        return []

def ddgs_news_search(query: str) -> list:
    try:
        with DDGS() as ddgs:
            return ddgs.news(query, max_results=5)
    except Exception as e:
        logger.error(f"Error in news search: {str(e)}")
        return []

news_search_tool = Tool(
    name="News Search",
    func=ddgs_news_search,
    description="Search the web for recent news articles."
)

text_search_tool = Tool(
    name="Text Search",
    func=ddgs_search,
    description="Search the web for financial data."
)

def browse_page_selenium(url: str, query: str) -> str:
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        content = driver.find_element("tag name", "body").text
        logger.info(f"Browsed {url}: {content[:200]}...")
        return content if content else "No content found"
    except Exception as e:
        logger.error(f"Error browsing {url}: {str(e)}")
        return f"Error browsing page: {str(e)}"
    finally:
        driver.quit()

browse_tool = StructuredTool.from_function(
    name="Browse Page",
    func=browse_page_selenium,
    description="Navigate to a URL and extract specific financial data or news."
)

# Define LangGraph nodes
def stock_data_node(state: FinanceState) -> FinanceState:
    if "stock_data" in state["plan"]:
        module = StockDataModule()
        result = module(state)
        state["results"]["stock_data"] = result["stock_data"]
        state["citations"].extend(result["sources"])
        logger.info(f"Stock data node result: {result}")
    return state

def fundamentals_node(state: FinanceState) -> FinanceState:
    if "fundamentals" in state["plan"]:
        module = FundamentalsModule()
        result = module(state)
        state["results"]["fundamentals_data"] = result["fundamentals_data"]
        state["citations"].extend(result["sources"])
        logger.info(f"Fundamentals node result: {result}")
    return state

def news_analysis_node(state: FinanceState) -> FinanceState:
    if "news_analysis" in state["plan"]:
        module = NewsAnalysisModule()
        result = module(state)
        state["results"]["news_summary"] = result["news_summary"]
        state["citations"].extend(result["sources"])
        logger.info(f"News analysis node result: {result}")
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
    seen = set()
    citations = [c for c in state["citations"] if not (c in seen or seen.add(c))]
    logger.info(f"Combined answer: {answer}, Citations: {citations}")
    return {"results": results, "answer": answer, "citations": citations}

# Build dynamic LangGraph workflow
def build_workflow(plan: List[str]) -> StateGraph:
    workflow = StateGraph(FinanceState)
    nodes = []
    for i, module in enumerate(plan):
        node_name = f"{module}_{i}"
        if module == "stock_data":
            workflow.add_node(node_name, stock_data_node)
            nodes.append(node_name)
        elif module == "fundamentals":
            workflow.add_node(node_name, fundamentals_node)
            nodes.append(node_name)
        elif module == "news_analysis":
            workflow.add_node(node_name, news_analysis_node)
            nodes.append(node_name)
    workflow.add_node("combine_results", combine_results_node)
    for i in range(len(nodes) - 1):
        workflow.add_edge(nodes[i], nodes[i + 1])
    if nodes:
        workflow.add_edge(nodes[-1], "combine_results")
        workflow.set_entry_point(nodes[0])
    else:
        workflow.set_entry_point("combine_results")
    workflow.set_finish_point("combine_results")
    logger.info(f"Workflow built with nodes: {nodes}")
    return workflow

# Run the pipeline
def run_pipeline(query: str) -> dict:
    logger.info(f"Starting pipeline for query: {query}")
    planner = Planner()
    plan = planner(query)
    logger.info(f"Generated Plan: {plan}")
    workflow = build_workflow(plan)
    state = FinanceState(
        query=query,
        plan=plan,
        results={},
        answer="",
        citations=[]
    )
    graph = workflow.compile()
    for step_state in graph.stream(state):
        logger.info(f"Pipeline step: {step_state}")
    final_state = step_state
    print(f"Final state: {final_state}") 
    return {"answer": final_state["combine_results"]["answer"], "citations": final_state["combine_results"]["citations"]}

# Example usage
if __name__ == "__main__":
    queries = [
        "What is Apple's current P/E ratio?",
    ]
    for query in queries:
        result = run_pipeline(query)
        print(f"Query: {query}")
        print(f"Answer: {result['answer']}")
        print(f"Citations: {result['citations']}\n")