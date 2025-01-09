from util import (
    get_pdf_collection,
    generate_response, get_answer, get_context,client, load_collection
)
import warnings
warnings.filterwarnings('ignore')
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import List, Dict, Any
import asyncio
import json
import logging
from transformers import logging as transformers_logging

# Silence transformers warning
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
async def chat_completion_request(messages: List[Dict], tools=None, model='gpt-4o-mini', stream=False) -> Any:
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
                    }
                },
                "required": ["query"],
            },
        },
    },
]

system_prompt = '''
Your task is to answer user questions. The user can ask a single question or multiple questions.
You have access to the get_context tool to return answers based on user single/multiple questions.
Return your thoughts if access database, ask user to wait if needed.
'''

async def main():
    db_status = False
    print('collection available: ', client.list_collections())
    collection_available = input('Is you collection available yes|no: ').strip()
    
    if collection_available.lower() == 'yes':
        collection_name = input('Please enter collection name: ').strip()
        status = load_collection(collection_name)
        
    if collection_available.lower() == 'no':
        pdf_path = input('Please provide pdf path to make database: ')
        print('Creating database...')
        status = get_pdf_collection(pdf_path)
    
    if status == 'success':
        while True:
            print("\n" + "-"*50)  # Separator line
            user_query = input('\nType q to quit or ask your query: ').strip()
            
            if user_query.lower() == 'q':
                print("\nGoodbye!")
                return
                
            if not user_query:
                print("Please enter a valid query.")
                continue
                
            system_msg = [{"role": "system", "content": system_prompt}]
            user_msg = [{"role": "user", "content": user_query}]
            feed_msg = system_msg + user_msg
            
            print("\nchatbot...")
            try:
                chat_answer = await chat_completion_request(feed_msg, tools=tools, stream=False)
                output_json = {}
                
                if hasattr(chat_answer, 'choices') and chat_answer.choices:
                    message = chat_answer.choices[0].message
                    
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        for call in message.tool_calls:
                            if call.function.name == 'get_answer':  # Corrected dot notation
                                arguments = json.loads(call.function.arguments)
                                query = arguments['query']
                                ans = await get_answer(query)
                                output_json[query] = ans['answer']
                        
                        if output_json:
                            print("\nGenerated JSON Output:")
                            print(json.dumps(output_json, indent=2))
                    elif hasattr(message, 'content') and message.content:
                        print(f"\nResponse: {message.content}")
                
            except Exception as e:
                print(f"\nError processing request: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user")
    except Exception as e:
        print(f"\nProgram terminated due to error: {str(e)}")
