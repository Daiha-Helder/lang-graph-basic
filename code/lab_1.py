from langchain_openai import ChatOpenAI
from langchain_core.messages import HumaMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os

# 1 - Carrega API Key
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# 2 - Definição de modelo
llm_model = ChatOpenAI(model = "gpt-3.5-turbo", 
                       api_key=API_KEY)

# 3 Define o prompt do sistema
system_message = SystemMessage(content = """
Você é um assistente. Se o usuário pedir contas, use 
a ferramenta 'somar'. Caso contrário, apenas responda normalmente.            
""")