from llm.gemini import get_feedback

def get_feedback_from_llm(hand_data: dict) -> str:
    return get_feedback(hand_data)
