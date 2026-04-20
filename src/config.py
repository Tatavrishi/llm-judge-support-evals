"""
Config module: loads environment variables and sets up the Groq client.
Every other file will import from here, so we only set up the API client once.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Check that your .env file exists "
        "and contains GROQ_API_KEY=your_key"
    )

# The Groq client — reused across the project
client = Groq(api_key=GROQ_API_KEY)

# Model choice: Llama 3.3 70B is Groq's flagship — fast and strong
# Free tier allows ~14,400 requests/day on this model
MODEL_NAME = "llama-3.3-70b-versatile"