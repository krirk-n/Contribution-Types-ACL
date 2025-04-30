import os
from openai import AzureOpenAI
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

    def get_chat_completion(self, messages, max_tokens=4096, temperature=1.0, top_p=1.0, model="gpt-4o"):
        if model == "o3-mini":
            response = self.client.chat.completions.create(
            messages=messages,
            max_completion_tokens=max_tokens,
            top_p=top_p,
            model=model
            )
        else:
            response = self.client.chat.completions.create(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            model=model
            )
        return response.choices[0].message.content

# Example usage
if __name__ == "__main__":
    azure_client = AzureOpenAIClient()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I am going to Paris, what should I see?"}
    ]
    response = azure_client.get_chat_completion(messages, model="o3-mini") # "gpt-4o" or "o3-mini" or "gpt-4o-mini"
    print(response)