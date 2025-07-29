# utils/formatting.py
def format_hand_for_llm(hand: dict) -> str:
    def card_str(card):
        return card if isinstance(card, str) else ' '.join(card)

    def format_street(street_data, name):
        out = [f"\n{name.capitalize()}:"]
        
        if 'card' in street_data:
            out.append(f"    Card: {street_data['card']}")
        if 'board' in street_data:
            out.append(f"    Board: {' '.join(street_data['board'])}")
        if 'pot' in street_data:
            out.append(f"    Pot: ${street_data['pot']}")
        if 'actions' in street_data:
            out.append("    Actions:")
            for act in street_data['actions']:
                out.append(f"        - {act['player']} {act['action']}")
        
        if 'metadata' in street_data and street_data['metadata']:
            meta = street_data['metadata']
            out.append("    Strategic Context:")
            if 'hero_position' in meta:
                out.append(f"        - Hero is in position: {meta['hero_position']}")
            if 'hero_action' in meta:
                out.append(f"        - Hero's action: {meta['hero_action']}")
            if 'facing_action' in meta:
                out.append(f"        - Faced: {meta['facing_action']}")
            if 'board_texture' in meta:
                out.append(f"        - Board texture: {meta['board_texture']}")
            if 'multiway' in meta:
                out.append(f"        - Multiway pot: {meta['multiway']}")
        
        if 'coaching' in street_data and street_data['coaching']:
            out.append("    Coaching Insight:")
            out.append(f"        - {street_data['coaching']}")
        
        return out

    lines = []

    # Summary
    lines.append("Hand Summary:")
    lines.append(f"- Hero: {hand['hero']['position']} with {' '.join(hand['hero']['hand'])}")

    if hand.get("positions"):  # backwards compatibility if positions/villains were used earlier
        lines.append("- Villains:")
        for v in hand['positions'].get('villains', []):
            lines.append(f"    - {v['position']} ( {v['description']} )")

    # Stakes + Stack sizes
    if hand.get("stakes"):
        lines.append(f"\nStakes: ${hand['stakes']} ({hand['players']} players)")

    if hand.get("stack_sizes"):
        lines.append("\nStack Sizes:")
        lines.append(f"    - Hero: ${hand['stack_sizes']['hero']}")
        for pos, size in hand['stack_sizes']['villains'].items():
            lines.append(f"    - {pos}: ${size}")

    if hand.get("blinds"):
        blinds = hand['blinds']
        lines.append("\nBlinds:")
        lines.append(f"    - SB: ${blinds['small']}")
        lines.append(f"    - BB: ${blinds['big']}")
        if blinds.get('straddle') != "None":
            lines.append(f"    - Straddle: ${blinds['straddle']}")

    # Streets
    lines.extend(format_street(hand['preflop'], "preflop"))
    lines.extend(format_street(hand['flop'], "flop"))
    lines.extend(format_street(hand['turn'], "turn"))
    lines.extend(format_street(hand['river'], "river"))

    # Result
    lines.append("\nShowdown:")
    if hand['result'].get('showdown_occurred'):
        lines.append(f"    - Hero: {hand['result']['hero_action']}")
        for v, cards in hand['result'].get('villain_showdowns', {}).items():
            lines.append(f"    - {v}: {' '.join(cards)}")
    else:
        lines.append(f"    - {hand['result']['hero_action']}")

    # Notes
    if hand.get("notes"):
        lines.append(f"\nNotes: {hand['notes']}")
    if hand.get("source"):
        lines.append(f"\nSource: {hand['source']}")

    return '\n'.join(lines)
