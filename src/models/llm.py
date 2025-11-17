from openai import OpenAI


def set_api_key(api_key: str):
    client = OpenAI(api_key=api_key)
    return client


def req_api(client, img_b64, img_format: str = 'jpeg', model: str = "gpt-5.1", system_msg: str = None, user_msg: str = None):
    if system_msg:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_msg},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{img_format};base64,{img_b64}"}
                        },
                    ],
                },
            ],
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
    else:
        return client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_msg},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{img_format};base64,{img_b64}"}
                        }
                    ]
                }
            ],
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )


def response_tokens_cost(response, price_per_token = 0.000002, price_completion = 0.000006):
  line_break = '\n========================'
  usage = response.usage
  prompt_tokens = usage.prompt_tokens
  completion_tokens = usage.completion_tokens
  total_tokens = usage.total_tokens

  print("Prompt tokens:", prompt_tokens)
  print("Completion tokens:", completion_tokens)
  print("Total tokens:", total_tokens, line_break)

  cost_prompt = prompt_tokens * price_per_token
  cost_completion = completion_tokens * price_completion
  cost_total = cost_prompt + cost_completion

  print("Cost prompt (USD):", cost_prompt)
  print("Cost completion (USD):", cost_completion, line_break)
  print("Cost total (USD):", cost_total, line_break, '\n')
