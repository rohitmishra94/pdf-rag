#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from openai import OpenAI, AsyncOpenAI
import os
from typing import List, Dict, Any, Optional
import logging.config
import json
import asyncio
from dotenv import load_dotenv
import tiktoken


print('env variable loaded: ',load_dotenv('/workspace/test/pdf-rag/env'))

logger = logging.getLogger(__name__)

AsyncClient = AsyncOpenAI()

def embedding_function_bge(text_list):
    return model.encode(text_list, return_dense=True)['dense_vecs']



class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = embedding_function_bge(input)
        return embeddings

model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True) 
default_ef = MyEmbeddingFunction()
client = chromadb.PersistentClient(path="chromadb_folder")


# In[2]:


# embedding_function_bge(['iam there'])
# default_ef(['iam there'])


# In[3]:


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

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:

    try:
        # Get the tokenizer for the specified model
        tokenizer = tiktoken.encoding_for_model(model)
    except KeyError:
        # Default to a generic encoding if the model is unknown
        tokenizer = tiktoken.get_encoding("cl100k_base")

    # Tokenize the text and return the token count
    token_count = len(tokenizer.encode(text))
    return token_count


# In[4]:


def create_pdf_collection(pdf_path):
    """
    Process a PDF file and add its chunks to a collection.
    
    Args:
        pdf_path: Path to the PDF file
        collection: The collection object to add documents to
    """
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
        status = 'success'
        return status,collection_name

    except Exception as e:
        logger.error(f"Error creating pdf collection: {str(e)}", exc_info=True)
        return f'Sorry for inconvenience. Error creating pdf collection. Please contact support.'


# In[5]:


# create_pdf_collection('/workspace/test/pdf-rag/handbook.pdf')


# In[6]:


def get_collections():
    return client.list_collections()

def load_collection(collection_name):
    try:
        collection = client.get_or_create_collection(name=collection_name,embedding_function=default_ef)
        status = 'success'
        return status, collection
    except Exception as e:
        logger.error(f"Collection load request failed: {str(e)}", exc_info=True)
        raise


# In[ ]:





# In[7]:


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


# In[8]:


async def generate_response(params: Dict[str, Any]) -> Any:
    """Generate response using OpenAI API with error handling and logging."""
    try:
        logger.info(f"Generating response with model: {params.get('model')}")
        response = await AsyncClient.chat.completions.create(**params)
        logger.info("Response generated successfully")
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        return f'Sorry for inconvenience. Please contact support.'

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
async def chat_completion_request(messages: List[Dict], model='gpt-4o-mini') -> Any:
    """Make a chat completion request with retry logic."""
    try:
        params = {
            'messages': messages,
            'max_tokens': 1000,
            'model': model,
            'temperature': 0,
            'response_format': {"type": "json_object"}
        }

        response = await generate_response(params)
        return response
    except Exception as e:
        logger.error(f"Chat completion request failed: {str(e)}", exc_info=True)
        return f'Sorry for inconvenience. Please contact support.'



# In[9]:


async def get_answer(query,collection_name):
    # context = get_context(query,n_results=5,top_results=2,threshold = 0.5)
    collection = client.get_or_create_collection(name=collection_name,embedding_function=default_ef)

    context, context_idx = get_full_context(query, collection, n_results=5, top=2)
    user_query = f'Based on below CONTEXT {context} ANSWER the query {query}'

    system_instruction = '''Ideal Output Format
The output should be a structured JSON blob that question with its corresponding answer.
Answers should be word to word match if the question is a word to word match
If the CONTEXT is insufficient, reply with “Data Not Available'''
    
    msg = [{"role": "system", "content": system_instruction},{"role": "user", "content": user_query}]
    
    response = await chat_completion_request(msg)
    total_tokens = response.usage.total_tokens
    output = json.loads(response.choices[0].message.content)

    return output,total_tokens


# In[22]:


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
async def chat_request(messages: List[Dict], tools=None, model='gpt-4o-mini', stream=False) -> Any:
    """Make a chat completion request with retry logic."""
    try:
        params = {
            'messages': messages,
            'max_tokens': 1500,
            'model': model,
            'temperature': 0,
            'tools': tools,
            'tool_choice': "auto",
            'stream': stream,
        }
        response = await generate_response(params)
        return response
    except Exception as e:
        logging.error(f"Chat completion request failed: {str(e)}", exc_info=True)
        raise

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_answer",
            "description": "Use this function to get answers based on context documents from database to user questions. The documents are purely based on user db.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "User query in string.",
                    },
                    "collection": {
                        "type": "string",
                        "description": "db collection name in string.",
                    }
                },
                "required": ["query","collection"],
            },
        },
    },

    
    {
        "type": "function",
        "function": {
            "name": "create_pdf_collection",
            "description": "Use this function to create database collection from pdf file path",
            "parameters": {
                "type": "object",
                "properties": {
                    "pdf_path": {
                        "type": "string",
                        "description": "pdf_path in string.",
                    }
                },
                "required": ["pdf_path"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_collections",
            "description": "Use this function to check exsiting collection in database",
            
        },
    },
]

system_prompt = '''
Your task is to answer user questions. The user can ask a single question or multiple questions.
You have access to the below tools to return answers based on user single/multiple questions.

# Tool
"name": "get_collections",
"description": "Use this function to check exsiting collection in database",
"name": "create_pdf_collection",
"description": "Use this function to create database collection from pdf file path",
"name": "get_answer",
"description": "Use this function to get answers based on context documents from database to user questions. The documents are purely based on user db.",
                     
# Flow of User interaction
greet the user in first interaction.
and introduce yourself --> you can only provide answers to question based on collection exists in database and list the collections name
using get_collection tool

if user collection does not exist ask user to provide pdf path and create collection based on pdf path using tool create_pdf_collection
pdf path name will be collection name to use. collection creation takes few seconds to ask user to wait few seconds.

if user provide collection name , that collection name will be used to answer user query using get_answer tool.
you can call get_answer tool multiple time for multiple questions

if user chat_history available then collection name based on latest chats will be used to answer query

YOUR ONLY TASK TO ANSWER USER QUERY BASED ON COLLECTION AVAILABLE ON DATABASE.
'''







async def chat_bot(chat_history, recursion_step=0):
    chat_answer = await chat_request(chat_history, tools=tools, stream=False)
    print('recursion step ', recursion_step)
    
    if hasattr(chat_answer, 'choices') and chat_answer.choices:
        message = chat_answer.choices[0].message
        if hasattr(message, 'content') and message.content:
            print(f"\nResponse: {message.content}")
            assistant_msg = [{"role": "assistant", "content": message.content}]
            chat_history += assistant_msg
            
        if hasattr(message, 'tool_calls') and message.tool_calls:
            output_json = {}
            needs_recursive_call = False
            
            for call in message.tool_calls:
                if call.function.name == 'get_answer':
                    print('searching answers...')
                    arguments = json.loads(call.function.arguments)
                    query = arguments['query']
                    collection_name = arguments['collection']
                    ans, _ = await get_answer(query, collection_name)
                    output_json[query] = ans['answer']
                    assistant_msg = [{"role": "assistant", "content": f"answer from database for {query}: {ans['answer']}"}]
                    chat_history += assistant_msg
                    
                elif call.function.name == 'get_collections':
                    print('fetching collection list...')
                    collection_list = get_collections()
                    if collection_list:
                        assistant_msg = [{"role": "assistant", "content": f'database collection list {collection_list}'}]
                    else: 
                        assistant_msg = [{"role": "assistant", "content": f'database collection list is empty'}]
                    chat_history += assistant_msg
                    needs_recursive_call = True
    
                elif call.function.name == 'create_pdf_collection':
                    print('creating collection...')
                    arguments = json.loads(call.function.arguments)
                    path = arguments['pdf_path']
                    status, collection_name = create_pdf_collection(path)
                    if status == 'success':
                        assistant_msg = [{"role": "assistant", "content": f'database collection created with collection name {collection_name}'}]
                    else:
                        assistant_msg = [{"role": "assistant", "content": f'database collection creation failed contact support'}]
                    chat_history += assistant_msg
                    needs_recursive_call = True
            
            if output_json:
                print("\nGenerated JSON Output:")
                print(json.dumps(output_json, indent=2))
            
            if needs_recursive_call:
                chat_answer = await chat_bot(chat_history, recursion_step+1)
    
    return chat_history


async def chat_bot_v2(chat_history, recursion_step=0, previous_action=None):
    print(f'recursion step {recursion_step}, previous action: {previous_action}')
    
    chat_answer = await chat_request(chat_history, tools=tools, stream=False)
    
    if hasattr(chat_answer, 'choices') and chat_answer.choices:
        message = chat_answer.choices[0].message
        if hasattr(message, 'content') and message.content:
            print(f"\nResponse: {message.content}")
            assistant_msg = [{"role": "assistant", "content": message.content}]
            chat_history += assistant_msg
            
        if hasattr(message, 'tool_calls') and message.tool_calls:
            output_json = {}
            current_action = None
            tool_calls_processed = False
            
            # First check if we have any get_answer calls
            has_get_answer = any(call.function.name == 'get_answer' for call in message.tool_calls)
            
            for call in message.tool_calls:
                # Handle get_answer differently - allow multiple in first pass
                if call.function.name == 'get_answer' and (previous_action != 'get_answer' or recursion_step == 0):
                    print('searching answers...')
                    arguments = json.loads(call.function.arguments)
                    query = arguments['query']
                    collection_name = arguments['collection']
                    ans, _ = await get_answer(query, collection_name)
                    output_json[query] = ans['answer']
                    assistant_msg = [{"role": "assistant", "content": f"answer from database for {query}: {ans['answer']}"}]
                    chat_history += assistant_msg
                    current_action = 'get_answer'
                    tool_calls_processed = True
                    
                # For other tools, keep the previous behavior
                elif call.function.name == 'get_collections' and previous_action != 'get_collections':
                    print('fetching collection list...')
                    collection_list = get_collections()
                    if collection_list:
                        assistant_msg = [{"role": "assistant", "content": f'database collection list {collection_list}'}]
                    else: 
                        assistant_msg = [{"role": "assistant", "content": f'database collection list is empty'}]
                    chat_history += assistant_msg
                    current_action = 'get_collections'
                    tool_calls_processed = True
                    break
    
                elif call.function.name == 'create_pdf_collection' and previous_action != 'create_pdf_collection':
                    print('creating collection...')
                    arguments = json.loads(call.function.arguments)
                    path = arguments['pdf_path']
                    status, collection_name = create_pdf_collection(path)
                    if status == 'success':
                        assistant_msg = [{"role": "assistant", "content": f'database collection created with collection name {collection_name}'}]
                    else:
                        assistant_msg = [{"role": "assistant", "content": f'database collection creation failed contact support'}]
                    chat_history += assistant_msg
                    current_action = 'create_pdf_collection'
                    tool_calls_processed = True
                    break
            
            if output_json:
                print("\nGenerated JSON Output:")
                print(json.dumps(output_json, indent=2))
            
            # Make recursive call only if:
            # 1. We processed some tool calls
            # 2. We're either in step 0 with get_answer calls or have a different action
            if tool_calls_processed and ((recursion_step == 0 and has_get_answer) or 
               (current_action and current_action != previous_action)):
                print(f'Making recursive call with previous_action: {current_action}')
                chat_answer = await chat_bot(chat_history, recursion_step + 1, current_action)
    
    return chat_history
  
  
        

# out = await chat_bot(chat_history)   

