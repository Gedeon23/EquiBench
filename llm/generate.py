# modified by github/Gedeon23: added support for ollama, generic openai style APIs and ScaDS.AI

# from litellm import completion
import os
import anthropic
import together
import openai

from steps.eval.type import EvalInput
from .utils import log_error_wrapper,  handle_error_openai, handle_error_anthropic, handle_error_together

# TODO
async def llm_generate(input: EvalInput) -> str:
    match input.model_platform:
        case "openai":
            return await generate_openai(input=input)
        case "anthropic":
            return await generate_anthropic(input=input)
        case "ollama":
            return await generate_ollama(input=input)
        case "together":
            return await generate_together(input=input)
        case "scads-ai":
            return await generate_scads_ai(input=input)
        case _:
            custom_api_type = os.environ.get(f"{input.model_platform.upper()}_API_TYPE")
            custom_api_base_url = os.environ.get(f"{input.model_platform.upper()}_API_BASE_URL")

            if not custom_api_type or not custom_api_base_url:
                raise RuntimeError(
f"""
{input.model_platform} is not a valid api provider. Change it or define type, base URL and API key for a custom api in your .env file:
<provider>_API_BASE_URL=<provider_base_url>
<provider>_API_KEY=<your_api_key>
<provider>_API_TYPE=openai
(currently only openai style APIs are supported)
"""
                )

            match custom_api_type:
                case "openai":
                    return await generate_generic_openai_api(input=input)
                case _:
                    raise RuntimeError(f"{custom_api_type} is not supported as a custom api type. Currently only openai style APIs are supported.")
            

@handle_error_openai
@log_error_wrapper
async def generate_openai(input: EvalInput) -> str:
    client = openai.AsyncClient()
    chat_completion = await client.chat.completions.create(
        messages = input.messages,
        model    = input.model_name,
        # timeout  = 60 * 5
    )
    return chat_completion.choices[0].message.content

@handle_error_openai
@log_error_wrapper
async def generate_scads_ai(input: EvalInput) -> str:
    api_key  = os.environ["SCADS_AI_API_KEY"]
    base_url = os.environ.get("SCADS_AI_BASE_URL", "https://llm.scads.ai/v1")
    client = openai.AsyncClient(api_key=api_key, base_url=base_url)
    chat_completion = await client.chat.completions.create(
        messages = input.messages,
        model    = input.model_name,
    )
    return chat_completion.choices[0].message.content

@handle_error_anthropic
@log_error_wrapper
async def generate_anthropic(input: EvalInput) -> str:
    client = anthropic.AsyncClient()
    message = await client.messages.create(
        max_tokens = 8192,
        model      = input.model_name,
        messages   = input.messages
    )
    return "".join([block.text for block in message.content])

@handle_error_openai
@log_error_wrapper
async def generate_ollama(input: EvalInput) -> str:
    client = openai.AsyncClient(
        base_url= os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1/"),
        api_key='ollama',
    )
    chat_completion = await client.chat.completions.create(
        messages = input.messages,
        model    = input.model_name,
    )
    return chat_completion.choices[0].message.content

@handle_error_openai
@log_error_wrapper
async def generate_generic_openai_api(input: EvalInput) -> str:
    client = openai.AsyncClient(
        base_url= os.environ.get(f"{input.model_platform.upper()}_API_BASE_URL"),
        api_key= os.environ.get(f"{input.model_platform.upper()}_API_KEY", "none"),
    )
    chat_completion = await client.chat.completions.create(
        messages = input.messages,
        model    = input.model_name,
    )
    return chat_completion.choices[0].message.content

@handle_error_together
@log_error_wrapper
async def generate_together(input: EvalInput) -> str:
    client = together.AsyncClient()
    chat_completion = await client.chat.completions.create(
        messages=input.messages,
        model   =f"{input.model_platform}/{input.model_name}" 
    )
    return chat_completion.choices[0].message.content

"""
@handle_error_deepseek
@log_error_wrapper
async def generate_deepseek(input: EvalInput) -> str:
    deepseek_api_key = os.environ["DEEPSEEK_API_KEY"]
    deepseek_base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    client = openai.AsyncClient(api_key=deepseek_api_key, base_url=deepseek_base_url)
    chat_completion = await client.chat.completions.create(
        messages = input.messages,
        model    = input.model_name,
        # timeout  = 60 * 5
    )
    return chat_completion.choices[0].message.content

#@handle_aisuide_error
@log_error_wrapper
@async_wrap
def create_completion_aisuite(input: EvalInput) -> str:
    client = aisuite.Client()
    chat_completion = client.chat.completions.create(
        messages = input.messages,
        model    = f"{input.model_platform}:{input.model_name}" 
    )
    return chat_completion.choices[0].message.content
"""
