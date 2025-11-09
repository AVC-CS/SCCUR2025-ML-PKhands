import random
import pandas as pd

# Poker labels (UCI order 0..9)
# 0: High Card, 1: One Pair, 2: Two Pair, 3: Three of a Kind,
# 4: Straight, 5: Flush, 6: Full House, 7: Four of a Kind,
# 8: Straight Flush, 9: Royal Flush

suits = [1, 2, 3, 4]
ranks = list(range(1, 14))  # 1..13

def make_high_card():
    while True:
        cards = random.sample([(s, r) for s in suits for r in ranks], 5)
        sset = {s for s, r in cards}
        rsorted = sorted(r for s, r in cards)
        is_flush = len(sset) == 1
        is_straight = rsorted == list(range(rsorted[0], rsorted[0] + 5)) or rsorted == [1,10,11,12,13]
        if not is_flush and not is_straight:
            return cards

def make_n_of_a_kind(kind):
    cards = []
    r_main = random.choice(ranks)
    if kind == 7:  # Four of a Kind
        cards = [(s, r_main) for s in random.sample(suits, 4)]
        r_extra = random.choice([r for r in ranks if r != r_main])
        cards.append((random.choice(suits), r_extra))
    elif kind == 6:  # Full House
        r2 = random.choice([r for r in ranks if r != r_main])
        cards = [(s, r_main) for s in random.sample(suits, 3)]
        cards += [(s, r2) for s in random.sample(suits, 2)]
    elif kind == 3:  # Three of a Kind
        cards = [(s, r_main) for s in random.sample(suits, 3)]
        extras = random.sample([r for r in ranks if r != r_main], 2)
        cards += [(random.choice(suits), extras[0]), (random.choice(suits), extras[1])]
    elif kind == 2:  # Two Pair
        r2 = random.choice([r for r in ranks if r != r_main])
        r3 = random.choice([r for r in ranks if r not in (r_main, r2)])
        cards = [(s, r_main) for s in random.sample(suits, 2)]
        cards += [(s, r2) for s in random.sample(suits, 2)]
        cards.append((random.choice(suits), r3))
    elif kind == 1:  # One Pair
        cards = [(s, r_main) for s in random.sample(suits, 2)]
        extras = random.sample([r for r in ranks if r != r_main], 3)
        cards += [(random.choice(suits), extras[0]), (random.choice(suits), extras[1]), (random.choice(suits), extras[2])]
    return cards

def make_straight(flush=False, royal=False):
    if royal:
        r_seq = [10, 11, 12, 13, 1]
    else:
        start = random.randint(1, 9)
        r_seq = list(range(start, start + 5))
    if flush:
        s = random.choice(suits)
        return [(s, r) for r in r_seq]
    else:
        return [(random.choice(suits), r) for r in r_seq]

def make_flush():
    s = random.choice(suits)
    while True:
        ranks5 = random.sample(ranks, 5)
        rsorted = sorted(ranks5)
        is_straight = rsorted == list(range(rsorted[0], rsorted[0]+5)) or rsorted == [1,10,11,12,13]
        if not is_straight:
            return [(s, r) for r in ranks5]

def make_hand_by_label(label):
    if label == 9: return make_straight(flush=True, royal=True)
    if label == 8: return make_straight(flush=True)
    if label == 7: return make_n_of_a_kind(7)
    if label == 6: return make_n_of_a_kind(6)
    if label == 5: return make_flush()
    if label == 4: return make_straight()
    if label == 3: return make_n_of_a_kind(3)
    if label == 2: return make_n_of_a_kind(2)
    if label == 1: return make_n_of_a_kind(1)
    if label == 0: return make_high_card()

def generate_balanced(samples_per_class=1000, out_csv="data/poker_balanced_10k.csv"):
    rows = []
    for label in range(10):
        for _ in range(samples_per_class):
            hand = make_hand_by_label(label)
            flat = [s for c in hand for s in c] + [label]
            rows.append(flat)
    df = pd.DataFrame(rows, columns=['S1','R1','S2','R2','S3','R3','S4','R4','S5','R5','ORD'])
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_csv}")

if __name__ == "__main__":
    generate_balanced()
