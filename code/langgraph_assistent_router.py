import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.router import workflow_router

def test_workflow():
    technician_result = workflow_router.invoke({
        "query": "Como posso aprender Python?"
    })
    
    print("\n=== Consulta Técnica ===")
    print(f"Pergunta: Como posso aprender Python?")
    print(f"Resposta: {technician_result['answer']}")

    health_result = workflow_router.invoke({
        "query": "Qual são os benefícios de uma alimentação saudável?"
    })
    
    print("\n=== Consulta Saúde ===")
    print(f"Pergunta: Qual são os benefícios de uma alimentação saudável?")
    print(f"Resposta: {health_result['answer']}")

    assistant_result = workflow_router.invoke({
        "query": "Qual é a capital do Brasil?"
    })
    
    print("\n=== Consulta Geral ===")
    print(f"Pergunta: Qual é a capital do Brasil?")
    print(f"Resposta: {assistant_result['answer']}")

if __name__ == "__main__":
    test_workflow()
