agent_prompt = """"
Você é um planejador de pesquisa.

Você está trabalhando em um projeto que visa responder às perguntas dos usuários usando fontes encontradas online.

Sua resposta deve ser técnica, utilizando informações atualizadas. 
Cite fatos, dados e informações específicas.

Aqui está a contribuição do usuário

<USER_INPUT>
{user_input}
</USER_INPUT>
"""

build_queries = agent_prompt + """
Seu primeiro objetivo é criar uma lista de consultas que serão usadas para encontrar respostas para a pergunta do usuário.

Responda com 3 a 5 consultas.
"""

resume_search = agent_prompt + """
Seu objetivo aqui é analisar os resultados da pesquisa na web e fazer uma síntese deles, enfatizando apenas o que é relevante para a pergunta do usuário.

Após o seu trabalho, outro agente usará a síntese para construir uma resposta final para o usuário, portanto certifique-se de que a síntese contenha apenas informações úteis e claras.

Seja conciso e claro.

Aqui estão os resultados da pesquisa na web:
<SEACH_RESULTS>
{search_results}
</SEACH_RESULTS>
"""

build_final_response = agent_prompt + """
Seu objetivo é desenvolver uma resposta final para o usuário usando os relatórios gerados durante a busca na web, com sua síntese.

A resposta deve conter entre 50 e 70 palavras.

Aqui estão os resultados da busca na web:
<SEACH_RESULTS>
{search_results}
</SEACH_RESULTS>
Você deve adicionar citações de referência (com o número da citação, exemplo [1]) para os artigos que você usou em cada parágrafo da sua resposta.
"""