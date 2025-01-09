import asyncio
import json
from typing import List, Dict
from util import system_prompt, chat_bot, count_tokens
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
async def run_chatbot():
    """
    Run an interactive command line chatbot that maintains conversation history
    and checks token limits.
    """
    try:
        chat_history: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        
        print("Chatbot initialized. Type 'q' to quit.")
        print("-" * 50)

        while True:
            user_input = input('Your input: ').strip()

            if user_input.lower() == 'q':
                print('Response: Goodbye!')
                break
                
            user_msg = [{"role": "user", "content": user_input}]
            chat_history.extend(user_msg)

            try:
                tokens = count_tokens(json.dumps(chat_history))
                if tokens > 100000:
                    print('Response: Context length exceeds limit. Quitting chat. Please start fresh!')
                    break
            except Exception as e:
                print(f'Error counting tokens: {str(e)}')
                break

            try:
                chat_history = await chat_bot(chat_history)
            except Exception as e:
                print(f'Error getting bot response: {str(e)}')
                break

    except KeyboardInterrupt:
        print('\nChat terminated by user.')
    except Exception as e:
        print(f'An unexpected error occurred: {str(e)}')

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(run_chatbot())