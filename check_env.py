import os
import importlib.metadata
from dotenv import load_dotenv

load_dotenv()

keys_to_check = ["GOOGLE_API_KEY", "LANGSMITH_API_KEY"]

for key in keys_to_check:
    value = os.getenv(key)
    if value is None:
        print(key, "-> MISSING")
    else:
        print(key, "-> set, length", len(value))

packages_to_check = ["langchain", "langchain-google-genai", "python-dotenv"]

for name in packages_to_check:
    print(name, "->", importlib.metadata.version(name))