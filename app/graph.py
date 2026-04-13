from pydantic import BaseModel
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from tavily import TavilyClient

from schemas import *
from prompts import *
import os
from dotenv import load_dotenv

OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 0 - Structured
class QueryList(BaseModel):
    queries: List[str]

# 1 - Modelos
llm = ChatOpenAI(
    model = "gpt-3.5-turbo",
    api_key = OPEN_API_KEY
)

reasoning_llm = ChatOpenAI(
    model = "o4-mini",
    api_key = OPEN_API_KEY
)


llms = ChatOllama(
    model="bjoernb/gemma4-e4b-fast",
    temperature=0.5
)

# 2 - Construção dos nós
def build_first_queries(state: ReportState):
    
    user_input = state.user_input
    prompt = build_first_queries.format(
        user_input = user_input
        )
    query_llm = llm.with_structured_output(QueryList)
    result = query_llm.invoke(prompt)

    return {"queries":result.queries}

def search_tavily(query: str):
    tavily_client = TavilyClient()

    results = tavily_client.search(
        query=query,
        max_results=1,
        include_raw_content=False
    )

    url = results["results"][0]["url"]
    url_extraction = tavily_client.extract(url)
    if (len(url_extraction['results'])>0):
        raw_content = url_extraction["results"][0]["raw_content"]



# 3 - Construção das arestas

# 4 - Construção do grafo 
builder = StateGraph(ReportState)
graph = builder.compile()

# Execução
if __name__ == "__main__":
    user_input = """
        Quero que você explique-me o processo total para construir um agente de IA.
    """
    graph.invoke({
        "user_input": user_input
    })