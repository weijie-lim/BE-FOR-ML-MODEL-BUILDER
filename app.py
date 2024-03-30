import asyncio
import json
import os
import requests

from constants import Constants
from datetime import datetime, date
from flask import Flask, jsonify, request, Response
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import VertexAIEmbeddings
# from langchain_google_vertexai import VertexAIEmbeddings
# from vertexai.language_models import TextEmbeddingModel
from rag import RAG
from upstash_redis import Redis
from prompts import RAG_PROMPT_TEMPLATE
# from state import State
from threading import Thread
from upstash_vector import Index
from upstash import UpstashVectorStore
from urllib.request import urlopen
from sentence_transformers import SentenceTransformer

#UPSTASH VECTOR
UPSTASH_URL = Constants.UPSTASH_URL
UPSTASH_TOKEN = Constants.UPSTASH_TOKEN
UPSTASH_DELETE_URL = Constants.UPSTASH_DELETE_URL

#REDIS
UPSTASH_REDIS_URL=Constants.UPSTASH_REDIS_URL
UPSTASH_REDIS_TOKEN=Constants.UPSTASH_REDIS_TOKEN
UPSTASH_REDIS_PASSWORD=Constants.UPSTASH_REDIS_PASSWORD
REDIS_KEY=Constants.REDIS_KEY

#HUGGING FACE INFERENCE ENDPOINT
HUGGING_FACE_ENDPOINT = Constants.HUGGING_FACE_ENDPOINT
HUGGING_FACE_TOKEN = Constants.HUGGING_FACE_TOKEN


app = Flask(__name__)

def queryHuggingFace(payload):
  headers = {
    "Accept" : "application/json",
    "Authorization": f"Bearer {HUGGING_FACE_TOKEN}",
    "Content-Type": "application/json" 
  }
  response = requests.post(HUGGING_FACE_ENDPOINT, headers=headers, json=payload)
  return response.json()

def getCall(prompt):
  responseJson = queryHuggingFace(prompt)
  return responseJson
  

def get_prompt(question, context):
    prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)
    return prompt

def format_and_update_qns_to_redis(question):
  try:
    redis = Redis(
      url=UPSTASH_REDIS_URL, 
      token=UPSTASH_REDIS_TOKEN)
    
    old_data = redis.get("source_of_truth")
    if old_data is None:
      old_data = ""
    '''
    Format:
    
    Source of Truth Context:
    Date of upload:
    Question:
    Answers:: XXXxxxx
    Date of upload:
    Question:
    Answers:: XXXxxxx
    '''
    new_data_str = '''
    Date of Question:{date}
    Question:{question}
    '''.format(
      date=date.today(),
      question=question
      )
    new_data_str = old_data + "\n" + new_data_str
    data = redis.set(REDIS_KEY, new_data_str)
    return "REDIS STORE UPDATED"
  except:
    return "FAILED TO UPDATE REDIS STORE"
    
def get_source_of_truth():
  # Include old source of truth into context
  try:
    redis = Redis(
      url=UPSTASH_REDIS_URL, 
      token=UPSTASH_REDIS_TOKEN)
    old_data = redis.get("source_of_truth")
    if old_data is None:
      old_data = ''
    return old_data
  except:
    return ""


@app.route('/')
def hello_world():
  return 'hello world'


@app.route('/submit_question_and_documents', methods=['POST'])
def submit_question_and_documents():
  try:
    req = request.get_json()
    # print(req.get('question'))
    # print(req.get('documents'))
    # print(req.get('autoApprove'))
    
    # try:
    #Get Document store from LangChain
    documents = []
    collated_dialogue_by_date = ""
    for url in req.get('documents'):
      page = urlopen(url)
      html_bytes = page.read()
      dialogue = html_bytes.decode("utf-8")
      
      date_str = url.split('/')[-1].split('_')[-2]
      date_format = '%Y%m%d'
      date_obj = datetime.strptime(date_str, date_format)
      
      curr_id = url.split('/')[-1].split('.')[0]
      date_string = date_obj.strftime("%Y-%m-%d")
      doc_name = url.split('/')[-1]
      
      documents.append(
        Document(
          page_content = dialogue,
          metadata = {
            'id': curr_id,
            'date_string': date_string,
            'doc_name': doc_name,
            'question': req.get('question')
          }
        )
      )
      
    
    #chunk document with chunk size of 1200 characters and a chunk_overlap of 200
    #before embedding, we need to chunk them to overcome LLM limitations in terms of input tokens and provides
    #fine grained info per chunk
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 1200,
      chunk_overlap = 200,
      separators=['.']
    )
    splits = text_splitter.split_documents(documents)
    index = Index(
      url=UPSTASH_URL,
      token=UPSTASH_TOKEN
    )
    model_name = 'all-MiniLM-L6-v2'
    encoder = SentenceTransformer(model_name)
    upstash_vector_store = UpstashVectorStore(index, encoder)
    ids = upstash_vector_store.add_documents(splits, batch_size=32)
    
    return Response(
        "Submitted vectors successfully",
        status=200,
    )
  except:
    return Response(
        "Submitted invalid key-value pairs",
        status=400,
    )

@app.route('/get_question_and_facts', methods=['GET'])
def get_question_and_facts():
  #get body
  try:
    index = Index(
      url=UPSTASH_URL,
      token=UPSTASH_TOKEN
    )
    
    # sentence transformer to embed
    model_name = 'all-MiniLM-L6-v2'
    encoder = SentenceTransformer(model_name)
    vectorstore = UpstashVectorStore(index, encoder)
    
    #get context and question
    query = ""
    results, list_of_ids = vectorstore.similarity_search_with_score(query)
    list_of_dates = []

    
    #get old source of truth if any
    old_data = get_source_of_truth()
    
    context = []
    for doc, _ in results:
      context.append(doc.page_content)
      list_of_dates.append(doc.metadata['date_string'])
      query = doc.metadata['question']
    
    
    # query = "What are our product design decisions?"
    responseData = {}
    responseData['question'] = query
    responseData['factsByDay'] = {}
    
    for c in range(len(context)):
      answers = []
      # few shot prompting does not work
      # prompt_to_give_1 = {
      #   'inputs': {
      #     'context': context[c],
      #     'question': '''
      #     Tell me fun facts about your hobbies?
      #     - crocheting.
      #     - scuba diving.
      #     - self-confessed Swiftie
          
      #     Tell me fun facts about recent events you have experienced?
      #     - listened to Nonsense by Sabrina Carpenter.
      #     - last vacation was my honeymoon.
      #     - I read was One Day by David Nicholls.
          
          
      #     {question}
      #     '''.format(question=query),
      #   },
      #   # 'parameters': {}
      # }
      # #use Hugging Face pretrained model
      # result_for_day = getCall(prompt_to_give_1)
      
      # try COT prompting
      
      # only for prompt 1, include old information into the model for chain of thought
      #other prompts are specifically about the current context
      prompt_to_give_1 = {
        'inputs': {
          'context': old_data + '\n' + context[c],
          'question': '''
          List out important information in the context.
          '''.format(question=query),
        },
        # 'parameters': {}
      }
      
      #use Hugging Face pretrained model
      result_for_day = getCall(prompt_to_give_1)
      
      prompt_to_give_2 = {
        'inputs': {
          'context': context[c],
          'question': '''
          Using the context, {question}
          '''.format(question=query),
        },
        # 'parameters': {}
      }
      #use Hugging Face pretrained model
      result_for_day = getCall(prompt_to_give_2)
      answers.append(result_for_day['answer'])
      
      prompt_to_give_3 = {
        'inputs': {
          'context': context[c],
          'question': '''
          Other key information in the context?
          '''.format(question=query),
        },
        # 'parameters': {}
      }
      #use Hugging Face pretrained model
      result_for_day = getCall(prompt_to_give_3)
      answers.append(result_for_day['answer'])
      
      prompt_to_give_4 = {
        'inputs': {
          'context': context[c],
          'question': '''
          Any more to add using the context?
          '''.format(question=query),
        },
        # 'parameters': {}
      }
      #use Hugging Face pretrained model
      result_for_day = getCall(prompt_to_give_4)
      answers.append(result_for_day['answer'])
      
      # add to return json
      responseData['factsByDay'][list_of_dates[c]] = answers
    
    responseData['status'] = 'done'  
    print(responseData)
    #upload to source_of_truth redis cache, the user will click yes or no for 
    #each fact on the UI
    result_of_redis = format_and_update_qns_to_redis(question=query)
    print(result_of_redis)
    
    #if all okay, delete all documents
    res = index.delete(list_of_ids)
    
    return Response(response=json.dumps(responseData), status=200)
  except:
    return Response(status=400)
  
if __name__ == '__main__':
  port = int(os.environ.get('PORT', 5000))
  app.run(debug=True, host='0.0.0.0', port=port)
  # app.run(debug=True)