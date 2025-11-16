# llm_handler.py
import os
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize API clients based on the active service
active_model_service = os.getenv("ACTIVE_MODEL_SERVICE", "openai")
client = None
genai = None

if active_model_service == 'openai':
    from openai import OpenAI
    if os.getenv("OPENAI_API_KEY"):
        client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
elif active_model_service == 'gemini':
    import google.generativeai as genai
    if os.getenv("GEMINI_API_KEY"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# from llama_vllm_api import get_response_llama_vllm_api

def get_llm_response(prompt_res, args):
    """
    Sends a prompt to the configured LLM and returns the parsed response.
    """
    model_engine = os.getenv("MODEL_ENGINE")
    if not model_engine:
        raise ValueError("MODEL_ENGINE environment variable not set in .env file")

    result = None
    raw_result = None
    
    try:
        if 'gpt' in model_engine:
            if active_model_service != 'openai' or client is None:
                raise ValueError("Cannot use GPT model. Check ACTIVE_MODEL_SERVICE and OPENAI_API_KEY in .env")
            
            test_chat_message = [{"role": "user", "content": prompt_res}]
            response = client.chat.completions.create(
                messages=test_chat_message, model=model_engine, temperature=0.7, max_tokens=2000
            )
            result = response.choices[0].message.content
            if 'gpt-4o' in model_engine:
                time.sleep(1)
                
        # elif 'Llama' in model_engine:
        #     result = get_response_llama_vllm_api(prompt_res)
            
        elif 'gemini' in model_engine:
            if active_model_service != 'gemini' or genai is None:
                raise ValueError("Cannot use Gemini model. Check ACTIVE_MODEL_SERVICE and GEMINI_API_KEY in .env")
            
            model = genai.GenerativeModel(model_engine)
            response = model.generate_content(prompt_res)
            result = response.text
            
        else: 
            raise ValueError(f"Unsupported model_engine: {model_engine}")
        
        raw_result = result
        # Clean and parse the JSON response
        result = result.replace('```', '').replace('json', '')
        parsed_result = json.loads(result)
        
        # Post-process the parsed result based on args
        if parsed_result.get('is_anomaly') and args.with_value:
            parsed_result['anomalies'] = [anomaly[0] for anomaly in parsed_result['anomalies']]
            
        return parsed_result, raw_result

    except Exception as e:
        print(f"Error in get_llm_response: {e}")
        if result:
            print(f"Failed to parse result: {result}")
        if 'gpt-4o' in model_engine:
            time.sleep(1)
        return None, None
