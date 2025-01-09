# from sentence_transformers import SentenceTransformer
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
print('env variable loaded: ',load_dotenv('env'))
logger = logging.getLogger(__name__)

AsyncClient = AsyncOpenAI()



def embedding_function_bge(text_list):
    return model.encode(text_list, return_dense=True)['dense_vecs']



class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = embedding_function_bge(input)
        return embeddings


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

def get_pdf_collection(pdf_path):
    """
    Process a PDF file and add its chunks to a collection.
    
    Args:
        pdf_path: Path to the PDF file
        collection: The collection object to add documents to
    """
    md_text = pymupdf4llm.to_markdown(pdf_path,show_progress=True)
    all_chunks = process_texts(md_text, chunk_size=500, overlap=50)
    
    global collection
    collection = client.get_or_create_collection(name=pdf_path,embedding_function=default_ef)
    print(collection)
    try:
        for idx, chunk in tqdm(enumerate(all_chunks)):
            id_ = str(idx)
            collection.add(
                documents=[chunk],
                ids=[id_]
            )

        return 'success'
    except Exception as e:
        logger.error(f"Error creating pdf collection: {str(e)}", exc_info=True)
        return f'Sorry for inconvenience. Please contact support.'

def get_unique_text_indices(text_list):
    unique_texts = {}
    unique_indices = []
    
    for i, text in enumerate(text_list):
        if text not in unique_texts:
            unique_texts[text] = i
            unique_indices.append(i)
    
    return unique_indices


def get_context(query,n_results=5,top_results=1,threshold = 0.5):
    result = collection.query(query_texts = query,n_results=n_results)
    ## reranker
    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
    texts = result['documents'][0]
    unique_indices = get_unique_text_indices(texts)
    pairs = [[query,texts[x]] for x in unique_indices]
    scores = reranker.compute_score(pairs, normalize=True)
    ## colbert
    query_col = model.encode([query],return_colbert_vecs=True)
    docs_col = model.encode(result['documents'][0],return_colbert_vecs=True)
    colber_scores = []
    for vectors in docs_col['colbert_vecs']:
        colber_scores.append(model.colbert_score(query_col['colbert_vecs'][0],vectors).numpy())

    ## combined+score
    all_score = [scores[i]+colber_scores[i] for i in range(len(scores))]

    valid_indices = [unique_indices[idx] for idx,i in enumerate(all_score) if i>threshold]
    data = [result['documents'][0][index] for index in valid_indices][:top_results]

    return data

def get_full_context(query, n_results=5, top=2):
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

    return top_context, top_ids


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
        raise



async def get_answer(query):
    # context = get_context(query,n_results=5,top_results=2,threshold = 0.5)
    context, context_idx = get_full_context(query, n_results=5, top=2)
    user_query = f'Based on below CONTEXT {context} ANSWER the query {query}'
    msg = [{"role": "system", "content": system_instruction},{"role": "user", "content": user_query}]
    
    response = await chat_completion_request(msg)
    output = json.loads(response.choices[0].message.content)

    return output

def load_collection(collection_name):
    try:
        global collection
        collection = client.get_or_create_collection(name=collection_name,embedding_function=default_ef)
        return 'success'
    except Exception as e:
        logger.error(f"Collection load request failed: {str(e)}", exc_info=True)
        raise

system_instruction = '''Ideal Output Format
The output should be a structured JSON blob that question with its corresponding answer.
Answers should be word to word match if the question is a word to word match
If the CONTEXT is insufficient, reply with “Data Not Available'''


model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True) 
default_ef = MyEmbeddingFunction()
client = chromadb.PersistentClient(path="chromadb_folder")
# chroma_client = chromadb.HttpClient(host='localhost', port=8000)
# collection = client.get_or_create_collection(name="db_v3",embedding_function=default_ef)
# reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)



