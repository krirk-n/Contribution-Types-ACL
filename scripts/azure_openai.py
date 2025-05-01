import os
import asyncio
import time
from openai import AzureOpenAI, RateLimitError, APIError, Timeout
from dotenv import load_dotenv

class AzureOpenAIClient:
    def __init__(self):
        load_dotenv()
        self.endpoint = os.getenv("AZURE_OPENAI_API_ENDPOINT")
        if self.endpoint is None:
            raise ValueError("AZURE_OPENAI_API_ENDPOINT environment variable not set.")

        self.subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
        if self.subscription_key is None:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable not set.")

        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        if self.api_version is None:
            raise ValueError("AZURE_OPENAI_API_VERSION environment variable not set.")

        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.subscription_key,
        )

    async def get_chat_completion(self, messages, max_tokens=4096, temperature=1.0, top_p=1.0,
                                  model="gpt-4o", retries=5):
        for attempt in range(retries):
            try:
                # Move blocking call to thread to avoid blocking asyncio loop
                if model == "o3-mini":
                    response = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        messages=messages,
                        max_completion_tokens=max_tokens,
                        top_p=top_p,
                        model=model
                    )
                else:
                    response = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        model=model
                    )
                
                return response.choices[0].message.content
            except (RateLimitError, Timeout, APIError) as e:
                wait_time = min(2 ** attempt, 30)
                print(f"[Retry {attempt+1}] Error: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

        raise RuntimeError("Max retries exceeded for chat completion.")

# Example usage
if __name__ == "__main__":
    async def main():
        azure_client = AzureOpenAIClient()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I am going to Paris, what should I see?"}
        ]
        response = await azure_client.get_chat_completion(messages, model="o3-mini")
        print(response)

    asyncio.run(main())
