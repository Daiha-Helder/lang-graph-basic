from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# 1 - Carrega API Key
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# 2 - Definição de modelo
llm_model = ChatOpenAI(model = "gpt-3.5-turbo", 
                       api_key=API_KEY)

# 3 - Define o estado do graph
class GraphState(BaseModel):
    input: str
    output: str
    tipo: str = None

# 4 - Função de realizar cáclulo
def realizar_calculo(state: GraphState) -> GraphState:
    return GraphState(input = state.input,
                      output = "Resposta de cáculo fictício: 42")

# 5 - Função para responder perguntas normais
def responder_curiosidade(state: GraphState) -> GraphState:
    response = llm_model.invoke([HumanMessage(content = state.input)])
    return GraphState(input = state.input,
                      output = response.content)

# 6 - Função para tratar perguntas não reconhecidas
def responder_erro(state: GraphState) -> GraphState:
    return GraphState(input = state.input,
                      output = "Desculpe, não entendi sua pergunta.")

# 7 - Função de classificação dos nodes
def classificar(state: GraphState) -> GraphState:
    pergunta = state.input.lower()
    if any(palavra in pergunta for palavra in ["soma", "quanto é", "+", "calcular"]):
        tipo = "calculo"
    elif any(palavra in pergunta for palavra in ["quem", "onde", "quando", "por que", "qual"]):
        tipo = "curiosidade"
    else:
        tipo = "desconhecido"
    return GraphState(input = state.input,
                      output = "",
                      tipo = tipo)

# 8 - Criando o Graph e Adcionando os Nodes
graph = StateGraph(GraphState)
graph.add_node("classificar", classificar)
graph.add_node("realizar_calculo", realizar_calculo)
graph.add_node("responder_curiosidade", responder_curiosidade)
graph.add_node("responder_erro", responder_erro)

# 9 - Adicionando condicionais
graph.add_conditional_edges(
    "classificar",
    lambda state: {
        "calculo": "realizar_calculo",
        "curiosidade": "responder_curiosidade",
        "desconhecido": "responder_erro"
    }[state.tipo]
)

# 10 - Definindo entrada e saída e compilação
graph.set_entry_point("classificar")
graph.set_finish_point(["realizar_calculo", "responder_curiosidade", "responder_erro"])
export_graph = graph.compile()

# 11 - Testando o Projeto
if __name__ == "__main__":
    exemplos = [
        "Quanto é 10 + 5",
        "Quem inventou a lâmpada",
        "Me diga um comando especial"
    ]

    for exemplo in exemplos:
        result = export_graph.invoke(GraphState(input = exemplo, output = ""))
        print(f"Pergunta: {exemplo}\nResposta: {result['output']}\n")