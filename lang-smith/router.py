from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage
from models import models

# 1 - System Messages
SYSTEM_MESSAGE_ASSISTANT = SystemMessage(
    content = """
    Você é um assistente virtual especializado em ajudar com diferentes tipos de consulta.
    Seja educado e prestativo sem suas respostas.
    """
)

SYSTEM_MESSAGE_TECHNICIAN = SystemMessage(
    content = """
    Você é um especialista técnico que fornece respostas detalhadas e precisas sobre tecnologia.
    Use linguagem técnica apropriada e forneça exemplos práticos quando possível.
    """
)

SYSTEM_MESSAGE_HEALTH = SystemMessage(
    content = """
    Você é um consultor de saúde que fornece informações gerais sobre bem estar e saúde.
    Lembre-se de sempre enfatizar que suas resposta são apenas informativas e não substituem consultas médicas.
    """
)

# 2 - Estados
class State(TypedDict):
    query: str
    category: str
    answer: str

def router(state: State):
    """Roteia a consulta para diferentes categorias baseado no conteúdo."""
    
    query = state['query'].lower()
    
    technician_words = [
        "python", "programação", 
        "código", "desenvolvimento", 
        "software", "tecnologia"
    ]

    health_words = [
        "saúde", "exercício",
        "alimentação", "bem-estar",
        "medicina", "dieta"
    ]

    if any(_words in query for _words in technician_words):
        return {"category":"technician"}
    elif any(_words in query for _words in health_words):
        return {"category":"health"}
    else:
        return {"category":"assistant"}

def assistant(state: State):
    """Processa consultas gerais"""
    
    messages = [
        SYSTEM_MESSAGE_ASSISTANT,
        HumanMessage(content=state['query'])
    ]

    response = models["gpt_3_5"].invoke(messages)
    return {"answer": response.content}

def technician(state: State):
    """Processa consultas técnicas"""
    
    messages = [
        SYSTEM_MESSAGE_TECHNICIAN,
        HumanMessage(content=state['query'])
    ]

    response = models["gpt_3_5"].invoke(messages)
    return {"answer": response.content}



def health(state: State):
    """Processa consultas sobre saúde"""
    
    messages = [
        SYSTEM_MESSAGE_HEALTH,
        HumanMessage(content=state['query'])
    ]

    response = models["gpt_3_5"].invoke(messages)
    return {"answer": response.content}

# 3 - Construindo o workflow
workflow_builder = StateGraph(State)

# Criando os nós
workflow_builder.add_node("router", router)
workflow_builder.add_node("assistant", assistant)
workflow_builder.add_node("technician", technician)
workflow_builder.add_node("health", health)

# Criando as arestas 
workflow_builder.set_entry_point("router")

workflow_builder.add_conditional_edges(
    "router",
    lambda state: state['category'], {
        "assistant": "assistant",
        "technician": "technician",
        "health": "health"
    }
)

workflow_builder.add_edge("assistant", END)
workflow_builder.add_edge("technician", END)
workflow_builder.add_edge("health", END)

workflow_router = workflow_builder.compile()


