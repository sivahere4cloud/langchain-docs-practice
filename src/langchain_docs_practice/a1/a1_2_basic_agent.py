from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()
MODEL = "google_genai:gemini-3.8-flash"


def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model=MODEL,
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in San Francisco?"}]}
)

for message in result["messages"]:
    print(type(message).__name__, "->", message.content)

print("FINAL ANSWER:", result["messages"][-1].content)