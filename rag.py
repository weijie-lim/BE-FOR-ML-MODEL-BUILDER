import os
from upstash_vector import Index
from prompts import RAG_PROMPT_TEMPLATE
from upstash import UpstashVectorStore


class RAG:
  def __init__(self, chat_box, embeddings):
    self.chat_box = chat_box
    self.set_llm()
    self.embeddings = embeddings
    self.index = Index(
        url=os.environ.get("UPSTASH_URL"),
        token=os.environ.get("UPSTASH_TOKEN"),
    )
    self.vectorstore = UpstashVectorStore(self.index, self.embeddings)

  def get_context(self, query):
    results = self.vectorstore.similarity_search_with_score(query)
    context = ""

    for doc, _ in results:
        context += doc.page_content + "\n===\n"
    return context, results

  @staticmethod
  def get_prompt(question, context):
    prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)
    return prompt

  def predict(self, query):
    context, source_documents = self.get_context(query)
    prompt = self.get_prompt(query, context)
    answer = self.llm.predict(prompt)
    prediction = {
        "answer": answer,
        "source_documents": source_documents,
    }
    return prediction