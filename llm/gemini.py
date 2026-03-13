import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-pro")

def get_feedback(hand: dict) -> str:
    # prompt = (
    #     "You are a professional poker coach. Your job is to give super in-depth, technical, strategic coaching "
    #     "on each street of the hand based on poker theory and common poker game strategies. You can reference concepts like protection, "
    #     "range advantage, GTO, exploitative adjustments, etc. The user has submitted a hand they played."
    #     "Here is the full hand history:\n\n{hand}\n\n"
    #     "Please give feedback on each street of the hand. Include what the user did well or poorly, "
    #     "and what action you would have taken instead (if different), and why. Be detailed but concise."
    # )
    prompt = ("""
              You are a professional poker coach who specializes in detailed hand reviews. 
              Your job is to give super in-depth, technical, strategic coaching
              on each street of the hand based on poker theory and common poker game strategies.

                Below is a hand history submitted by a student. For each street, give feedback on the hero's decision:
                - Was the play optimal?
                - What other options could be considered?
                - What can be inferred from villain actions?
                - What are the villain's likely ranges?
                - What possible hands could the villain have?
                - How does my specific hand interact with the board?
                - What would you have done differently, and why?

                Always consider:
                - Effective stacks
                - Board texture
                - Villain tendencies
                - Pot odds, ranges, and blockers
                - Exploitative vs. GTO deviations

                Do not assume incorrect hole cards or pot sizes.

                Here is the hand:

                {formatted_hand_goes_here}
              """
)

    response = model.generate_content(prompt)
    return response.text.strip()