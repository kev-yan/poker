# utils/formatting.py

def format_hand_for_llm(hand: dict) -> str:
    def card_str(card):
        return card if isinstance(card, str) else ' '.join(card)

    lines = []

    lines.append("Hand Summary:")
    lines.append(f"- Hero: {hand['positions']['hero']} with {' '.join(hand['hero_hand'])}")
    lines.append("- Villains:")
    for v in hand['positions']['villains']:
        lines.append(f"    - {v['position']} ( {v['description']} )")

    lines.append("\nStack Sizes:")
    lines.append(f"    - Hero: ${hand['stack_sizes']['hero']}")
    for pos, size in hand['stack_sizes']['villains'].items():
        lines.append(f"    - {pos}: ${size}")

    blinds = hand['blinds']
    lines.append("\nBlinds:")
    lines.append(f"    - SB: ${blinds['small']}")
    lines.append(f"    - BB: ${blinds['big']}")
    if blinds.get('straddle') != "None":
        lines.append(f"    - Straddle: ${blinds['straddle']}")

    lines.append("\nPreflop:")
    for act in hand['preflop']['actions']:
        lines.append(f"    - {act['player']} {act['action']}")
    lines.append(f"    Pot: ${hand['preflop']['pot']}")

    lines.append(f"\nFlop: {' '.join(hand['flop']['board'])}")
    for act in hand['flop']['actions']:
        lines.append(f"    - {act['player']} {act['action']}")
    lines.append(f"    Pot: ${hand['flop']['pot']}")

    lines.append(f"\nTurn: {hand['turn']['card']}")
    for act in hand['turn']['actions']:
        lines.append(f"    - {act['player']} {act['action']}")

    lines.append(f"\nRiver: {hand['river']['card']}")
    for act in hand['river']['actions']:
        lines.append(f"    - {act['player']} {act['action']}")

    lines.append("\nShowdown:")
    if hand['result'].get('showdown_occurred') == True:
        lines.append(f"    - Hero: {hand['result']['hero_action']}")
        for v, cards in hand['result'].get('villain_showdowns', {}).items():
            lines.append(f"    - {v}: {' '.join(cards)}")
    else:
        lines.append(f"    - {hand['result']['hero_action']}")

    lines.append(f"\nNotes: {hand.get('notes', 'None')}")

    return '\n'.join(lines)
