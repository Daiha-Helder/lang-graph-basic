from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
import os 
from dotenv import load_dotenv

# 1 - Configurações iniciais 
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

model = ChatOpenAI(
    model = "o4-mini",
    api_key = API_KEY
)


# 2 - Prompt do Sistema (o agente)
system_message = SystemMessage(
    content = """
        Você é um pesquisador muito sarcástico e irônico.
        Use ferramenta 'search' sempre que necessário, especialmente
        para perguntas que exigem informações da web.
        """
)

# 3 - Criando a ferramenta search
@tool('search')
def search_web(query: str = "") -> str:
    """
    Busca informações na web baseada na consulta fornecida.

    Args:
        query: Termo para buscar dados na web

    Returns:
        As informações encontradas na web ou uma mensagem indicando
        que nenhuma informação foi encontrada.
    """

    tavily_search = TavilySearchResults(max_results = 3)
    search_docs = tavily_search.invoke(query)

    return search_docs

# 4 - Criação do agente ReAct
tools = [search_web]
graph = create_react_agent(
    model = model,
    tools = tools,
    prompt = system_message
)

export_graph = graph