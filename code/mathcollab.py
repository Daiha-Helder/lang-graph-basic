import json
from typing import TypedDict, List, Dict, Any
import re

# 1 - Estado Compartilhado
class SimpleAgentState(TypedDict):
    original_query: str
    plan: List[Dict[str, str]] | None
    current_task_idx: int
    intermediate_results: Dict[str, str]
    final_response: str | None
    error: str | None
    current_task_description: str | None
    current_specialist_type: str | None
    current_task_id: str | None
    specialist_result: str | None

# 2 - Construção dos nós 

# 2.1 - Planejador simples divide em cálculo e explicação
def planner_node(state: SimpleAgentState) -> Dict[str,str]:
    query = state['original_query']
    plan = [
        {
            "task_id": "do_math",
            "specialist_type":"mathematician",
            "description": f"Resolva a seguinte expressão matemática: {query}"
        },
        {
            "task_id": "explain_result",
            "specialist_type": "write",
            "description": f"Explique-me em linguagem simples o resultado do cálculo feito na tarefa 'do_math'"

        }
    ]
    return {
        "plan": plan,
        "current_task_idx": 0,
        "intermediate_results": {},
        "error": None,
        "specialist_type": None
    }

# 2.2 - Prepara para próxima tarefa
def prepare_next_task_node(state: SimpleAgentState) -> Dict[str, Any]:
    plan = state.get("plan", [])
    current_task_idx = state.get("current_task_idx", 0)

    if not plan or current_task_idx >= len(plan):
        return {
            "current_task_id": None,
            "current_task_description": None,
            "current_specialist_type": None
        }

    current_task = plan[current_task_idx]
    return {
            "current_task_id": current_task["task_id"],
            "current_task_description": current_task["description"],
            "current_specialist_type": current_task["specialist_type"]
        }

# 2.3 - Resolve a equação matemática
def mathematician_node(state: SimpleAgentState) -> Dict[str, str]:
    desc = state.get("current_task_description", "")

    matchs = re.search(r":\s*(.+)", desc)
    if not matchs:
        return {
            "specialist_result":"Não foi possível identificar uma expressão matemática"
        }
    expr = matchs.group(1).strip()
    try:
        result = eval(expr)
        return  {"specialist_result": str(result)}
    except Exception as e:
        return {"specialist_result": f"Erro ao calcular: {e}"}

# 2.4 - Escreve o resultado 
def writer_node(state: SimpleAgentState) -> Dict[str, str]:
    
    prev_results = state.get("intermediate_results", {})
    math_result = prev_results.get("do_math", "sem resultado")

    return {
        "specialist_result": f"O resultado do cálculo é {math_result}. Isso significa que, ao resolver a expressão chegamos a esse valor."
    }

# 2.5 - Coleta o resultado e avança no nó
def collect_result_and_advance_node(state: SimpleAgentState) -> Dict[str, str]:
    current_task_id = state.get("current_task_id")
    
    specialist_output = state.get(
        "specialist_result",
        "Nenhum resultado do especialista encontrado no estado."
    )

    updated_intermediate_results = state.get(
        "intermediate_result",
         {}
    ).copy()

    if current_task_id:
        updated_intermediate_results[current_task_id] = specialist_output
    
    new_idx = state.get(
        "current_task_idx",
        0) + 1
    
    return {
        "intermediate_result": updated_intermediate_results,
        "current_task_idx": new_idx,
        "specialist_result": None
    }

# 2.6 - Sintetiza a informação
def systhesis_node(state: SimpleAgentState) -> Dict[str, str | None]:
    original_query = state["original_query"]

    intermediate_results = state.get(
        "intermediate_results",
        {}
    )

    response = f"Pergunta original: {original_query}\n"
    for task_id, result in intermediate_results.items():
        response += f"-{task_id}: {result}\n"
    return {
        "final_response": response,
        "error": None
    }

# 2.7 - Executa a tarefa ou sitentiza
def should_execute_task_or_synthesize(state: SimpleAgentState) -> str:
    
    if state.get("error"):
        return "error_handler"
    
    plan = state.get("plan", [])
    current_task_idx = state.get("current_task_idx", 0)

    if current_task_idx < len(plan):
        return "prepare_next_task"
    else:
        return "systhesize_response"

# 2.8 - Roteia os nós
def specialist_router_node(state: SimpleAgentState) -> str:
    
    specialist_type = state.get("current_specialist_type")
    if specialist_type == 'mathematician':
        return 'mathematician'
    elif specialist_type == "writer":
        return "writer"
    else:
        return "error_handler"

# 2.9 - Retorna o erro
def error_node(state: SimpleAgentState) -> Dict[str, str | None]:
    error_message = state.get("error", "Erro desconhecido no workflow")
    return {
        "final_response": f"Ocorreu um erro: {error_message}"
    }