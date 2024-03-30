RAG_PROMPT_TEMPLATE = """
Your task is to answer questions by using a given context.

Don't invent anything that is outside of the context.
Write your Answers in 5 point forms

%CONTEXT%
{context}

%Question%
{question}

Hint: Do not copy the context. Use your own words.

Answer:
"""