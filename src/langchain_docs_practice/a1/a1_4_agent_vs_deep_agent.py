import truststore

truststore.inject_into_ssl()

import urllib.request
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

load_dotenv()

MODEL = "google_genai:gemini-3.8-flash"
BOOK_URL = "https://www.gutenberg.org/files/64317/64317-0.txt"

SYSTEM_PROMPT = """You are a careful data assistant.
Use tools to get real data. Do not estimate or guess numbers.
If you cannot compute an exact answer, say so."""

TASK = f"Fetch {BOOK_URL} and tell me exactly how many lines contain the word 'Gatsby'."


def fetch_text_from_url(url: str) -> str:
    """Download a document and return its full text, or an error message."""
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            raw_bytes = response.read()
        return raw_bytes.decode("utf-8")
    except Exception as error:
        return f"Error fetching {url}: {error}"


model = init_chat_model(MODEL, temperature=0, timeout=120)

plain_agent = create_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
)

deep_agent = create_deep_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
)

message = {"messages": [{"role": "user", "content": TASK}]}

print("=== PLAIN AGENT ===")
plain_result = plain_agent.invoke(message)
print(plain_result["messages"][-1].text)

print("=== DEEP AGENT ===")
deep_result = deep_agent.invoke(message)

for item in deep_result["messages"]:
    for call in getattr(item, "tool_calls", []):
        print("TOOL CALL:", call["name"])

print(deep_result["messages"][-1].text)