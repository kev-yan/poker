import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-pro")

def get_feedback(hand: dict) -> str:
    prompt = (
        "You are a professional poker coach. Your job is to give technical, strategic coaching "
        "on each street of the hand based on poker theory and common poker game strategies. You can reference concepts like protection, "
        "range advantage, GTO, exploitative adjustments, etc. The user has submitted a hand they played."
        "Here is the full hand history:\n\n{hand}\n\n"
        "Please give feedback on each street of the hand. Include what the user did well or poorly, "
        "and what action you would have taken instead (if different), and why. Be detailed but concise."
    )

    response = model.generate_content(prompt)
    return response.text.strip()