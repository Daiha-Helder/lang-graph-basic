import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.news_collab_llm import news_workflow, NewsAgentState

def main():
    print("=== Multiagente de Análise de Notícia ===")
    news = input("Digite o texto da notícia (ou cole o conteúdo): ")

    # 1 - Estado incial
    state = NewsAgentState(
        original_news = news,
        plan = None,
        current_task_idx = 0,
        intermediate_results = {},
        final_response = None,
        error = None,
        current_task_description = None,
        current_specialist_type = None,
        current_task_id = None,
        specialist_result = None
    )

    # 2 - Executa o workflow
    result = news_workflow.invoke(state)
    print("\n=== Resposta final ===")
    print(result.get("final_response", "Nenhuma resposta gerada."))

if __name__ == "__main__":
    main()