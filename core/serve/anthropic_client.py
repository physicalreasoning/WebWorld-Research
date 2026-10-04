import os
import time
import anthropic

client = anthropic.Anthropic(
    api_key=os.environ.get('ANTHROPIC_API_KEY'),
)


def call_api(model, messages, max_retries=30, **kwargs):
    if 'max_tokens' not in kwargs:
        kwargs['max_tokens'] = 8192

    # Anthropic expects system message separately
    system = None
    filtered_messages = []
    for msg in messages:
        if msg['role'] == 'system':
            system = msg['content']
        else:
            filtered_messages.append(msg)

    for attempt in range(max_retries):
        try:
            api_kwargs = {
                'model': model,
                'messages': filtered_messages,
                **kwargs,
            }
            if system:
                api_kwargs['system'] = system

            response = client.messages.create(**api_kwargs)
            return response.content[0].text

        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}, retrying...")
            time.sleep(1)

    raise Exception("Max retries failed")


if __name__ == "__main__":
    print("Testing Anthropic API call")

    test_messages = [
        {"role": "user", "content": "Hello, this is a test message. Please reply in one sentence."}
    ]

    try:
        response = call_api("claude-sonnet-4-20250514", test_messages, max_retries=3)
        print("API call succeeded!")
        print("Response:", response)
    except Exception as e:
        print(f"API call failed: {e}")
