# -*- coding: utf-8 -*-
"""fully-agentic-flow.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16KFto6AffISrovcGDFMVRknTq4GfLeZ9
"""

!pip install -r /content/requirements.txt

from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import json
from dotenv import load_dotenv
import tiktoken


print('env variable loaded: ',load_dotenv('/content/env'))

openai_client = OpenAI()

class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o-mini"
    instructions: str = "You are a helpful Agent"
    tools: list = []

class Response(BaseModel):
    agent: Optional[Agent]
    messages: list


def execute_tool_call(tool_call, tools, agent_name):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"{agent_name}:", f"{name}({args})")

    return tools[name](**args)

import inspect

def function_to_schema(func) -> dict:
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }

def sample_function(param_1, param_2, the_third_one: int, some_optional="John Doe"):
    """
    This is my docstring. Call this function when you want.
    """
    print("Hello, world")

schema =  function_to_schema(sample_function)
print(json.dumps(schema, indent=2))

from chromadb.utils import embedding_functions
from FlagEmbedding import FlagReranker
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from FlagEmbedding import BGEM3FlagModel
import pymupdf4llm
from tqdm import tqdm
import numpy as np
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
# from openai import OpenAI, AsyncOpenAI
import os
from typing import List, Dict, Any, Optional
import logging.config
import json
import asyncio
import tiktoken


logger = logging.getLogger(__name__)

def embedding_function_bge(text_list):
    return model.encode(text_list, return_dense=True)['dense_vecs']



class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = embedding_function_bge(input)
        return embeddings

model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True)
default_ef = MyEmbeddingFunction()
client = chromadb.PersistentClient(path="chromadb_folder")

def generate_response(params: Dict[str, Any]) -> Any:
    """Generate response using OpenAI API with error handling and logging."""
    try:
        logger.info(f"Generating response with model: {params.get('model')}")
        response = openai_client.chat.completions.create(**params)
        logger.info("Response generated successfully")
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        return f'Sorry for inconvenience. Please contact support.'

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages: List[Dict], model='gpt-4o-mini') -> Any:
    """Make a chat completion request with retry logic."""
    try:
        params = {
            'messages': messages,
            'max_tokens': 1000,
            'model': model,
            'temperature': 0,
            'response_format': {"type": "json_object"}
        }

        response = generate_response(params)
        return response
    except Exception as e:
        logger.error(f"Chat completion request failed: {str(e)}", exc_info=True)
        return f'Sorry for inconvenience. Please contact support.'

import fitz
check_pdf_prompt = ''' Analyze the page content and return True if page as table of content information.
return json output {'toc': true or false}. pdf_page is
'''
def check_pdf_page_for_index(pdf_path):
    '''
    Read the pdf from the pdf path and return the text file path containing index of the pdf if index found.
    '''

    scanned_pages_dict = {}
    doc = fitz.open(pdf_path)
    doc_name = doc.name.split('/')[-1]
    # Iterate through each page
    print('checking for index..')
    for page_num in range(10):
        page = doc.load_page(page_num)  # Load the current page
        text = page.get_text()
        msg = [{"role": "system", "content": check_pdf_prompt + f'{text}'}]
        response = chat_completion_request(msg)
        output = json.loads(response.choices[0].message.content)
        scanned_pages_dict[page_num] = output['toc']

    index_pages = [k for k,v in scanned_pages_dict.items() if v==True]
    index_text = [doc.load_page(i).get_text() for i in index_pages]
    if index_text:
        collection_name = pdf_path#.split('/')[-1]
        with open(f'{collection_name}_index.txt','w') as f:
            f.write(pdf_path+'\n')
            f.write('\n'.join(index_text))
            # print('index saved at ',f'{collection_name}_index.txt')
        # return f'{collection_name}_index.txt'
        return f"index saved at f'{collection_name}_index.txt"
    else:
      return f"no index found for {pdf_path}"

def check_if_index_exist(pdf_path):
  '''
  Check if index of pdf file exist or not.
  '''
  file_path = f'{pdf_path}_index.txt'
  if os.path.exists(file_path):
    return f'index found at {file_path}'
  else:
    return f'index not found at {file_path}'

def read_index_file(index_file_path):
  '''
  Read the index file and return the text.
  '''
  with open(index_file_path, 'r') as f:
    lines = f.readlines()
    return '\n'.join(lines[:])

def get_answer(query,context):
  '''
  Input: query, context
  Return the answer based on context for given query.
  '''
  system_instruction = ''' Return json blob with question with its corresponding answer.
  Answers should be word to word match if the question is a word to word match
  If the CONTEXT is insufficient, reply with “Data Not Available'''

  user_query = f'Based on below CONTEXT {context} ANSWER the query {query}'
  msg = [{"role": "system", "content": system_instruction},{"role": "user", "content": user_query}]

  response = chat_completion_request(msg)
  print('got response')
  total_tokens = response.usage.total_tokens
  output = json.loads(response.choices[0].message.content)['answer']

  return output

def get_context_based_on_index(query,index_file_path):
  '''
  Input: query, index_file_path
  Return the context based on index of pdf file for given query.
  '''


  with open(f'{index_file_path}','r') as f:
    pdf_path = f.readline().strip()
    content = f.read()
    prompt = f'''
  Your task is to return list of page no which may contain information regarding {query}
  based on this table of content {content} output format: json page:[page no]
  '''
  msg = [{"role": "system", "content": prompt}]
  response =chat_completion_request(msg)
  output = json.loads(response.choices[0].message.content)

  doc = fitz.open(pdf_path)
  print('looking for context at pages: ', output['page'])
  context = [doc.load_page(page_num-1).get_text() for page_num in output['page']]
  if context:
    print('context found')
    return ' '.join(context)
  else:
    return f'No context found for given {query} through index {index_file_path} of file'




def get_answer_based_on_index_context(query,index_file_path):
  '''
  Input: query, context based on index of file
  Return the answer based on context for given query.
  '''

  context = get_context_based_on_index(query,index_file_path)
  answer = get_answer(query,context)
  return answer

def process_texts(texts, chunk_size=100, overlap=30):
    """Process a list of texts, splitting them into chunks of specified size with overlap,
    and accumulating shorter texts."""
    accumulated_words = []  # Accumulate words from texts shorter than chunk_size
    final_chunks = []  # Store the final chunks of text

    for text in texts.split():
        accumulated_words.append(text)

        while len(accumulated_words) >= chunk_size:
            # Take the first chunk_size words for the current chunk
            chunk = " ".join(accumulated_words[:chunk_size])
            final_chunks.append(chunk)
            # Remove words from the start of the accumulated_words, considering overlap
            accumulated_words = accumulated_words[chunk_size - overlap:]

    # If there are any remaining words, form the last chunk
    if accumulated_words:
        final_chunks.append(" ".join(accumulated_words))

    return final_chunks


def get_unique_text_indices(text_list):
    unique_texts = {}
    unique_indices = []

    for i, text in enumerate(text_list):
        if text not in unique_texts:
            unique_texts[text] = i
            unique_indices.append(i)

    return unique_indices


def create_db_collection_from_pdf(pdf_path):
  '''
  Create collection of pdf file.
  '''
  try:
        md_text = pymupdf4llm.to_markdown(pdf_path,show_progress=True)
        all_chunks = process_texts(md_text, chunk_size=500, overlap=50)

        collection_name = pdf_path.split('/')[-1]
        collection = client.get_or_create_collection(name=collection_name,embedding_function=default_ef)
        logger.info(collection)

        for idx, chunk in tqdm(enumerate(all_chunks)):
            id_ = str(idx)
            collection.add(
                documents=[chunk],
                ids=[id_]
            )
        return f'data processed and saved in db with collection name {collection_name}'

  except Exception as e:
      logger.error(f"Error creating pdf collection: {str(e)}", exc_info=True)
      return f'Sorry for inconvenience. Error creating pdf collection. Please contact support.'

def get_collections():
  '''
  Return collections list if exists
  '''
  collection_list =  client.list_collections()
  if collection_list:
    return ' , '.join(collection_list)
  else:
    return f'No collections found'


def get_full_context(query, collection, n_results=5, top=2):
    logger.info(f'quering collection---> {collection}')

    result = collection.query(query_texts = query,n_results=n_results)
    texts = result['documents'][0]
    ids = result['ids'][0]
    unique_indices = get_unique_text_indices(texts)
    unique_docs = [texts[x] for x in unique_indices]
    unique_ids = [ids[x] for x in unique_indices]
    ## colbert
    query_col = model.encode([query],return_colbert_vecs=True)
    docs_col = model.encode(unique_docs,return_colbert_vecs=True)
    colber_scores = []
    for vectors in docs_col['colbert_vecs']:
        colber_scores.append(model.colbert_score(query_col['colbert_vecs'][0],vectors).numpy())

    ## full_context_colbert
    full_context_scores = []
    full_context_ids = []
    for id in unique_ids:
        pre_id,post_id = str(int(id)-1), str(int(id)+1)
        # print(pre_id,id,post_id)
        full_context_ids.append([pre_id,id,post_id])
        full_context=collection.get(ids=[f'{pre_id}',f'{id}',f'{post_id}'])['documents']
        full_context = ''.join(full_context)
        full_context_colber_vec = model.encode([full_context],return_colbert_vecs=True)
        full_context_colber_score = model.colbert_score(query_col['colbert_vecs'][0],full_context_colber_vec['colbert_vecs'][0]).numpy()

        full_context_scores.append(full_context_colber_score)

    all_scores = [2*full_context_scores[i]+0.9*colber_scores[i] for i in range(len(colber_scores))]
    sorted_indices = [index for index, _ in sorted(enumerate(all_scores), key=lambda x: x[1], reverse=True)]
    top_context_ids_list = [full_context_ids[index] for index in sorted_indices][:top]
    flattened_list = np.array(top_context_ids_list).flatten().tolist()
    top_ids = list(set(flattened_list))
    top_context = collection.get(ids=top_ids)['documents']

    logger.info(f'context retrieved from collection---> {collection}')
    return top_context, top_ids


def get_context_based_on_collection(query,collection_name):
  '''
  Input: query, collection name
  Return the context based on collection of pdf file for given query.
  '''
  collection = client.get_or_create_collection(name=collection_name,embedding_function=default_ef)

  context, context_idx = get_full_context(query, collection, n_results=5, top=2)

  if context:
    print('context found')
    return ' '.join(context)
  else:
    return f'No context found for given {query} through collection {collection_name}'

def get_answer_based_on_collection(query,collection_name):
  '''
  Input: query, context based on index of file
  Return the answer based on context for given query.
  '''

  context = get_context_based_on_collection(query,collection_name)
  answer = get_answer(query,context)
  return answer







def transfer_to_index_agent():
  '''
  Return the index based answerin agent.

  '''
  return index_agent

def tranfer_to_collection_agent():
  '''
  Return the collection based answering agent.

  '''
  return collection_agent

def transfer_to_answer_agent():
  '''
  Return the answer based on index or collection agent.

  '''
  return answer_agent


def transfer_to_main_agent():
  '''
  Return the main agent.

  '''
  return main_agent


index_agent = Agent(
    name="Index Agent",
    model="gpt-4o-mini",
    instructions='''
    You are a helpful Agent. Your task is to provide answer based on pdf path provide by user.
    You have access to below tools:
    check_if_index_exist: use this to check index file exist or not,
    check_pdf_page_for_index: use this to extract index of pdf file.
    transfer_to_index_agent: use this to transfer the query to index based answerin agent if index exists.

    workflow:
    check if index exist or not,
    if not try to create index, if index created then transfer to index based answerin agent.
    ''',
    tools = [check_pdf_page_for_index,check_if_index_exist,transfer_to_answer_agent]
)

collection_agent = Agent(
    name="Collection Agent",
    model="gpt-4o-mini",
    instructions='''
    You are a helpful Agent. Your task is to provide answer based on pdf path provide by user.
    You have access to below tools:
    get_collections: use this to get existing collection list,
    create_db_collection_from_pdf: use this to create collection from pdf path.
    transfer_to_collection_agent: use this to transfer the query to collection based answerin agent if collection exists.

    workflow:
    check if collection exist or not,
    if not try to create collection, if collection created then transfer to collection based answerin agent.

    ''',
    tools = [get_collections,create_db_collection_from_pdf,transfer_to_answer_agent]
)


answer_agent = Agent(
    name="Answering Agent",
    model="gpt-4o-mini",
    instructions='''
    You are a helpful Agent. Your task is to answer the of user query based on index or collection of pdf file.
    if index/collection based answer not found sufficient back to main agent''',
    tools = [get_answer_based_on_index_context, get_answer_based_on_collection, transfer_to_main_agent]

)




main_agent = Agent(
    name="Main Agent",
    model="gpt-4o-mini",
    instructions='''
    You are a helpful Agent. Your task is to provide answer based on pdf path provide by user.
    You have access to below tools:
    transfer_to_index_agent: use this to transfer the query to index based answerin agent if index exists.
    transfer_to_collection_agent: use this to transfer the query to collection based answerin agent if collection exists.

    workflow:
    always go for index based anwer route to answer query ,if sufficent answer not found then try for collection based answer route.

    ''',
    tools = [transfer_to_index_agent,tranfer_to_collection_agent]
)

def run_full_turn(agent, messages):

    current_agent = agent
    num_init_messages = len(messages)
    messages = messages.copy()

    while True:

        # turn python functions into tools and save a reverse map
        tool_schemas = [function_to_schema(tool) for tool in current_agent.tools]
        tools = {tool.__name__: tool for tool in current_agent.tools}

        # === 1. get openai completion ===
        response = openai_client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "system", "content": current_agent.instructions}]
            + messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        messages.append(message)

        if message.content:  # print agent response
            print(f"{current_agent.name}:", message.content)

        if not message.tool_calls:  # if finished handling tool calls, break
            break

        # === 2. handle tool calls ===

        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools, current_agent.name)

            if type(result) is Agent:  # if agent transfer, update current agent
                current_agent = result
                result = (
                    f"Transfered to {current_agent.name}. Adopt persona immediately."
                )

            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)

    # ==== 3. return last agent used and new messages =====
    return Response(agent=current_agent, messages=messages[num_init_messages:])


def execute_tool_call(tool_call, tools, agent_name):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"{agent_name}:", f"{name}({args})")

    return tools[name](**args)  # call corresponding function with provided arguments

agent = main_agent
messages = []

while True:
    user = input("User: ").strip()
    if user.lower()=='q':
      break
    messages.append({"role": "user", "content": user})

    response = run_full_turn(agent, messages)
    agent = response.agent
    messages.extend(response.messages)





# pip install pymupdf
