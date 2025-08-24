import os
import openai
import time
from openai import OpenAI
client = OpenAI()

# 1) Configure your API key (make sure OPENAI_API_KEY is set in your env)
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# 2) Define the two agents' system prompts
agent_a_system = (
    "You are Agent A, a friendly AI enthusiastic about chatting on many topics. "
    "When you receive something from Agent B, respond informally and add new insights."
)

agent_b_system = (
    "You are Agent B, a knowledgeable AI who enjoys detailed discussions. "
    "When Agent A speaks, respond thoughtfully and ask follow-up questions."
)

# 3) Initialize each agent's conversation history
history_a = [{"role": "system", "content": agent_a_system}]
history_b = [{"role": "system", "content": agent_b_system}]

# 4) Helper to get a chat response
def get_chat_response(messages, model="gpt-3.5-turbo", temperature=0.8):
    # OpenAI uses `openai.Completion.create` in the updated API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=150  # Limit to prevent overly long responses
    )
    return response.choices[0].message.content.strip()

# 5) Define the topics and run the back-and-forth
topics = ["cricket", "weather", "political news"]

for topic in topics:
    # Agent A starts the topic
    prompt_a = f"Hey Agent B, let's chat about {topic}!"
    history_a.append({"role": "user", "content": prompt_a})
    reply_a = get_chat_response(history_a)
    history_a.append({"role": "assistant", "content": reply_a})

    # Pass A's reply to Agent B
    history_b.append({"role": "user", "content": reply_a})
    reply_b = get_chat_response(history_b)
    history_b.append({"role": "assistant", "content": reply_b})

    # (Optionally) pass B's reply back to A for one more turn
    history_a.append({"role": "user", "content": reply_b})
    reply_a2 = get_chat_response(history_a)
    history_a.append({"role": "assistant", "content": reply_a2})

    # Print out the mini-dialogue for this topic
    print(f"\n=== Conversation on {topic.upper()} ===")
    print(f"A ▶ {prompt_a}")
    print(f"A ◀ {reply_a}")
    print(f"B ▶ {reply_a}")
    print(f"B ◀ {reply_b}")
    print(f"A ▶ {reply_b}")
    print(f"A ◀ {reply_a2}")

    # Small pause to avoid rate limits
    time.sleep(1)