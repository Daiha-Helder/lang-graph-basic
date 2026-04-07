from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langchain.messages import SystemMessage, HumanMessage
from src.models import models

# 1 - System Message
SYSTEM_MESSAGE_LLMS = SystemMessage(
    content = """
    Você é um especialista em análise de código e boas práticas de programação.
    Sua tarefa é analisar o código fornecido e sugerir melhorias em termos de:

    1. Performance e otimização
    2. Boas práticas e padrões de código
    3. Segurança e tratamento de erros
    4. Legibilidade e manutenibilidade

    Forneça suas sugestões de forma estrutura e clara, com exemplos práticos de como 
    implementar as melhorias sugeridas.
    Seja específico e detalhado em suas recomendações.
    """
)

# 2 - Definição do Estado
class State(TypedDict):
    query: str # Código a ser analisado
    llm1: str # Análise do Gemini
    llm2: str # Análise do o4 mini
    best_llm: str # Retorna a melhor análise escolhida

# 3 - Construção dos nós
def call_llm_1(state: State):
    "Recebe o código e retorna a análise do modelo gpt 3.5"
    messages = [
        SystemMessage(content = SYSTEM_MESSAGE_LLMS.content),
        HumanMessage(content = f"Analise o seguinte código e forneça susgestões de melhorias: \n\n\{state['query']}")
    ]

    response = models['gpt_3_5'].invoke(messages)
    return {'llm1': response.content}

def call_llm_2(state: State):
    "Recebe o código e retorna a análise do modelo o4 mini"
    messages = [
        SystemMessage(content = SYSTEM_MESSAGE_LLMS.content),
        HumanMessage(content = f"Analise o seguinte código e forneça susgestões de melhorias: \n\n\{state['query']}")
    ]

    response = models['o4'].invoke(messages)
    return {'llm2': response.content}

def judge(state: State):
    """Avalia qual análise foi mais completa e útil"""
    
    msg = f"""
    Atue como revisor técnico sênior e avalie a qualidade das análises de código fornecidas por dois especialistas.

    Sua tarefa é escolher a análise que:
    1. Identifica mais problemas potenciais.
    2. Fornece sugestões mais práticas e implementáveis.
    3. Considere aspectos do código, como performance, segurança, legibilidade, etc.
    4. Explica melhor o raciocínio por trás das sugestões.
    
    [Código Analisado]
    {state['query']}

    [Análise de Especialista A]
    {state['llm1']}

    [Análise de Especialista B]
    {state['llm2']}

    Forneça sua avaliação comparativa e conclua com seu veredito final usando exatamente um destes formatos:
    '[[A]] se a análise A for melhor'
    '[[B]] se a análise B for melhor'
    '[[C]] em caso de empate'
    """

    messages = [
        SystemMessage(content=msg)
    ]

    response = models['gpt_4o'].invoke(messages)
    return {"best_llm": response.content}

# 4 - Construindo o Workflow
code_analysis_builder = StateGraph(State)

# Adicionando os nós 
code_analysis_builder.add_node("call_llm_1", call_llm_1)
code_analysis_builder.add_node("call_llm_2", call_llm_2)
code_analysis_builder.add_node("judge", judge)


# Adiciona as arestas
code_analysis_builder.add_edge(START, "call_llm_1")
code_analysis_builder.add_edge(START, "call_llm_2")
code_analysis_builder.add_edge("call_llm_1", "judge")
code_analysis_builder.add_edge("call_llm_2", "judge")
code_analysis_builder.add_edge("judge", END)

# Construindo Graph
code_analysis_workflow = code_analysis_builder.compile()