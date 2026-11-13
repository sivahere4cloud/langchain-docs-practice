from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()

MODEL = "google_genai:gemini-3.8-flash"

def get_order_status(order_id: str) -> str:
    """Look up the shipping status of a customer order by its order id."""
    orders = {
         "A100": "Shipped, arriving Friday",
        "A101": "Processing, not shipped yet",
    }
    return orders.get(order_id,f"No order found with id {order_id}")



def get_refund_policy() -> str:
    """Return the company refund policy."""
    return "Refunds are allowed within 30 days of delivery, if the item is unused."

agent = create_agent(
    model=MODEL,
    tools=[get_order_status, get_refund_policy],
    system_prompt=(
        "You are a support assistant for an online shop. "
        "Use tools for facts about orders and policies. Be concise."
    ),
    name="Customer Support Agent",
)    


result = agent.invoke(
    {
        "messages":[
            {
                "role": "user",
                 "content": "Where is order A100, and can I get a refund on it?",
            }
        ]
    }
)


for message in result["messages"]:
    names = [c["name"] for c in getattr(message, "tool_calls", [])]
    print(type(message).__name__, names, message.text[:100])
    if type(message).__name__ == "AIMessage":
        print("RAW tool_calls:", message.tool_calls)


print("\nFINAL:", result["messages"][-1].text)