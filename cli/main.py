import json
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path

def get_hand_input():
    print("Enter your hand as a list of cards (e.g., ['2H', '3D', '4C']):")
    hand = {"stake" : input("Enter the blind level in $ (e.g., 1/2 or 0.25/0.5): "),
            "stack_size": input("Effective stack size ($): "),
            "hand": input("Your hand (e.g., AhKh): "),
            "preflop_action": input("Preflop action: "),
            "flop": input("Flop cards: "),
            "flop_action": input("Flop action: "),
            "turn": input("Turn card: "),
            "turn_action": input("Turn action: "),
            "river": input("River card: "),
            "river_action": input("River action: "),
            "villain_description": input("Villain description: "),
            "notes": input("Any other notes about the hand: ")
        }
    return hand


