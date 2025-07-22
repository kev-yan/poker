# gpt/prompts.py

import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_coaching_feedback(hand_data):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional poker coach. Your job is to give technical, strategic coaching "
                "on each street of the hand based on poker theory and common poker game strategies. You can reference concepts like protection, "
                "range advantage, GTO, exploitative adjustments, etc. The user has submitted a hand they played."
            )
        },
        {
            "role": "user",
            "content": f"Here is the full hand history:\n\n{hand_data}\n\n"
                       "Please give feedback on each street of the hand. Include what the user did well or poorly, "
                       "and what action you would have taken instead (if different), and why. Be detailed but concise."
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7
    )

    return response["choices"][0]["message"]["content"]
