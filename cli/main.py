import json
from pathlib import Path
from utils.io_helpers import load_data, save_data
from utils.format import format_hand_for_llm
from llm.prompts import generate_coaching_feedback
from llm.coach_api import get_feedback_from_llm

DATA_PATH = Path("data/sample_hand.json")

def display_hand_summary(hand):
    print("\n--- Poker Hand Summary ---")

    print(f"Hero Position: {hand['positions']['hero']}")
    print(f"Hero Hand: {', '.join(hand['hero_hand'])}")
    print("Villains:")
    for v in hand['positions']['villains']:
        print(f"  - {v['position']}: {v['description']}")

    print(f"\nBlinds: SB = {hand['blinds']['small']}, BB = {hand['blinds']['big']}, Straddle = {hand['blinds'].get('straddle')}")
    print(f"Stack Sizes:")
    print(f"  Hero: {hand['stack_sizes']['hero']}")
    for pos, size in hand['stack_sizes']['villains'].items():
        print(f"  {pos}: {size}")

    print("\nPreflop:")
    for a in hand['preflop']['actions']:
        print(f"  {a['player']}: {a['action']}")
    print(f"  Pot: {hand['preflop']['pot']}")

    print("\nFlop:")
    print(f"  Board: {' '.join(hand['flop']['board'])}")
    for a in hand['flop']['actions']:
        print(f"  {a['player']}: {a['action']}")
    print(f"  Pot: {hand['flop']['pot']}")

    print("\nTurn:")
    print(f"  Card: {hand['turn']['card']}")
    for a in hand['turn']['actions']:
        print(f"  {a['player']}: {a['action']}")

    print("\nRiver:")
    print(f"  Card: {hand['river']['card']}")
    for a in hand['river']['actions']:
        print(f"  {a['player']}: {a['action']}")

    print("\nResult:")
    
    print(f"  Hero Action: {hand['result']['hero_action']}")
    for villain, cards in hand['result']['villain_showdowns'].items():
        print(f"  {villain}: {', '.join(cards)}")

    print("\nNotes:")
    print(f"  {hand['notes']}")

def main():
    print("Welcome to the Poker Coach CLI")
    while True:
        print("\nOptions:")
        print("1. View Sample Hand")
        print("2. Input New Hand Data")
        print("3. Exit")

        choice = input("\nChoose an option (1-3): ")
        if choice == '1':
            try:
                hand = load_data(DATA_PATH)
                formatted_hand = format_hand_for_llm(hand)
                print("\nHand Breakdown:")
                print(formatted_hand)

                print("Send this hand to the Poker Coach for feedback? (y/n): ")
                if input().strip().lower() == 'y':
                    feedback = get_feedback_from_llm(formatted_hand)
                    print("\n--- Coaching Feedback ---")
                    print(feedback)
            except FileNotFoundError:
                print("Sample hand data not found. Please ensure the file exists.")
            except json.JSONDecodeError:
                print("Error decoding the sample hand data. Please check the file format.")
        elif choice == '2':
            print("This feature is not yet implemented. Please check back later.")
        elif choice == '3':
            print("Exiting the Poker Coach CLI. Goodbye!")
            break
        else:
            print("Invalid option. Please choose a valid option (1-3).")

if __name__ == "__main__":
    main()