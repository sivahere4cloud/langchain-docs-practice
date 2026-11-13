from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent

load_dotenv()

MODEL = "google_genai:gemini-3.8-flash"


def get_order_status(order_id: str) -> str:
    """Look up the shipping status of a customer order by its order id."""
    orders = {
        "A100": "Shipped, arriving Friday",
        "A101": "Processing, not shipped yet",
    }
    return orders.get(order_id, f"No order found with id {order_id}")


# My example schema, not from the docs. The docs use their own `Answer` model.
class OrderAnswer(BaseModel):
    """The agent's final answer about an order."""

    order_id: str = Field(description="The order id the user asked about")
    status: str = Field(description="Shipping status, taken only from the tool result")
    found: bool = Field(description="False if no order was found")


agent = create_agent(
    model=MODEL,
    tools=[get_order_status],
    system_prompt="You are a support assistant. Only state facts returned by tools.",
    response_format=OrderAnswer,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the status of order A101?"}]}
)

answer = result["structured_response"]
print(type(answer).__name__)
print(answer)
print(answer.order_id, answer.status, answer.found)

#print("\nstate keys:", list(result.keys()))

print("\nRAW RESULT:", result)