# modified by github/Gedeon23: add support for generic openai style APIs

import os
import asyncio
from typing    import Optional
from anthropic import APIConnectionError, InternalServerError, RateLimitError
from openai    import AsyncOpenAI
from pydantic  import BaseModel

class CodeEquivalentResult(BaseModel):
    equivalent: Optional[bool]
    
async def judge_openai(content: str, model_name: str="gpt-4o-2024-11-20"):
    client = AsyncOpenAI()
    while True:
        try:
            completion = await client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": "Extract the code equivalent answer from text, the text answer YES means equivalent, answer NO means inequivalent. True for equivalent, False for not quivalent. If it is not decidable, return None. "},
                {"role": "user", "content": content},
            ],
                response_format=CodeEquivalentResult,    
            )
            result = completion.choices[0].message.parsed
            return result.equivalent
        except (RateLimitError, InternalServerError, APIConnectionError) as error:
            match error.code:
                case "rate_limit_exceeded":
                    # Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-4o in organization org-tHX7813yeiJY2uTj13V8QSmP on tokens per min (TPM): Limit 30000, Used 29929, Requested 72. Please try again in 2ms. Visit https://platform.openai.com/account/rate-limits to learn more.', 'type': 'tokens', 'param': None, 'code': 'rate_limit_exceeded'}}
                    if error.message.find("tokens per min (TPM)") != -1:
                        await asyncio.sleep(60)
                    else:
                        await asyncio.sleep(1)
                    continue
                case "insufficient_quota":
                    raise error
                case _:
                    raise error

async def judge_generic_openai_api(content: str, model_platform: str, model_name: str):
    client = AsyncOpenAI(
        base_url=os.environ.get(f"{model_platform.upper()}_API_BASE_URL"),
        api_key=os.environ.get(f"{model_platform.upper()}_API_KEY"),
    )
    while True:
        try:
            completion = await client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Extract the code equivalent answer from text, the text answer YES means equivalent, answer NO means inequivalent. True for equivalent, False for not quivalent. If it is not decidable, return None. "},
                    {"role": "user", "content": content},
                ],
                response_format=CodeEquivalentResult,    
            )
            result = completion.choices[0].message.parsed
            return result.equivalent
        except (RateLimitError, InternalServerError, APIConnectionError) as error:
            match error.code:
                case "rate_limit_exceeded":
                    # Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-4o in organization org-tHX7813yeiJY2uTj13V8QSmP on tokens per min (TPM): Limit 30000, Used 29929, Requested 72. Please try again in 2ms. Visit https://platform.openai.com/account/rate-limits to learn more.', 'type': 'tokens', 'param': None, 'code': 'rate_limit_exceeded'}}
                    if error.message.find("tokens per min (TPM)") != -1:
                        await asyncio.sleep(60)
                    else:
                        await asyncio.sleep(1)
                    continue
                case "insufficient_quota":
                    raise error
                case _:
                    raise error

async def llm_judge(content: str):
    model_with_platform = os.environ.get("JUDGE_MODEL")
    if not model_with_platform:
        raise RuntimeError(f"please set a model for judging malformatted classifications in your .env file: JUDGE_MODEL=<provider/model_name>")
    model_with_platform = model_with_platform.split("/", maxsplit=1)
    model_platform = model_with_platform[0]
    model_name = model_with_platform[1]

    match model_platform:
        case "openai":
            judge_api = lambda content: judge_openai(content, model_name=model_name)
        case _:
            custom_api_type = os.environ.get(f"{model_platform.upper()}_API_TYPE")
            custom_api_base_url = os.environ.get(f"{model_platform.upper()}_API_BASE_URL")

            if not custom_api_type or not custom_api_base_url:
                raise RuntimeError(
f"""
{model_platform} is not a valid api provider. Change it or define type, base URL and API key for a custom api in your .env file:
<provider>_API_BASE_URL=<provider_base_url>
<provider>_API_KEY=<your_api_key>
<provider>_API_TYPE=openai
(currently only openai style APIs are supported)
"""
                )

            match custom_api_type:
                case "openai":
                    judge_api = lambda content: judge_generic_openai_api(content, model_name=model_name, model_platform=model_platform)
                case _:
                    raise RuntimeError(f"{custom_api_type} is not supported as a custom api type. Currently only openai style APIs are supported.")


    content = content.strip()
    sentences = [sentence.strip() for sentence in content.rstrip(".").split(".")]

    if not sentences:
        return None
    
    # Check the last two sentence
    last_sentence = sentences[-1]
    result = await judge_api(content=last_sentence)
    if result is not None:
        return result
    
    if len(sentences) >= 2:
        last_two_sentence = sentences[-2:]
        result = await judge_api(content=".".join(last_two_sentence))
        if result is not None:
            return result
    
    result = await judge_api(content=content)
    return result
