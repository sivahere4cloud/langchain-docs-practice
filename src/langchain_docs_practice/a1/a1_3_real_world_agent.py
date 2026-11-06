
import truststore

truststore.inject_into_ssl()
import urllib.request
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

MODEL = "google_genai:gemini-3.8-flash"
MAX_CHARS = 5000

SYSTEM_PROMPT = """You are a careful web page assistant.
Always use the fetch_text_from_url tool to read a page before answering.
If the tool returns an error, say so. Do not guess page content."""


def fetch_text_from_url(url: str) -> str:
    """Download a web page and return its text, or an error message."""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            raw_bytes = response.read()
        text = raw_bytes.decode("utf-8")
        return text[:MAX_CHARS]
    except Exception as error:
        return f"Error fetching {url}: {error}"


model = init_chat_model(MODEL, temperature=0, timeout=30)
checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "a1-3-demo"}}

first = agent.invoke(
    {"messages": [{"role": "user", "content": "Fetch https://example.com and tell me in one sentence what it says."}]},
    config,
)
print("FIRST:", first["messages"][-1].content)

second = agent.invoke(
    {"messages": [{"role": "user", "content": "Which URL did I ask you to fetch?"}]},
    config,
)
print("SECOND:", second["messages"][-1].content)