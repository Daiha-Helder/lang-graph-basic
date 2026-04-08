from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from pydantic import BaseModel
from langchain_core.runnables.graph import MermaidDrawMethod
from dotenv import load_dotenv
import os

# 1 - Carrega API Key
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# 2 - Definição de modelo
llm_model = ChatOpenAI(model = "gpt-3.5-turbo", 
                       api_key=API_KEY)


# 3 - Definição do StateGraph
class GraphState(BaseModel):
    input: str
    output: str

# 4 - Função de resposta
def responder(state):
    response = llm_model.invoke([HumanMessage(content=state.input)])
    return GraphState(input = state.input, 
                      output = response.content)

# 5 - Criando  o Graph
graph = StateGraph(GraphState)
graph.add_node("responder", responder)
graph.set_entry_point("responder")
graph.set_finish_point("responder")

# 6 - Compilando o Grafo 
export_graph = graph.compile()

# 7 - Gerando a imagem .png do grafo
png_bytes = export_graph.get_graph().draw_mermaid_png(
    draw_method = MermaidDrawMethod.API
)


# 7 - Testando o Agente
if __name__ == "__main__":
    result = export_graph.invoke(GraphState(input="Quem descobriu a América?",
                                            output=""))
    
    print(result)

   
    # 1 - Visualizar o grafo - primeira forma, usando o site "mermaid.live para interpretar
    # print(export_graph.get_graph().draw_mermaid()) 

    # 2 - Visualizar o grafo - segunda forma
    with open("./data/grafo_exemplo_0.png", "wb") as f:
        f.write(png_bytes)
