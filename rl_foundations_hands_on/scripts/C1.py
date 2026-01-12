import random

# Toy retail environment
# State 0 = "normal customers"
# State 1 = "discount-seeking customers" (they wait unless you discount)
#
# Action 0 = no promotion
# Action 1 = run promotion

def step(state, action):
    # Reward = profit (higher is better)
    # Transitions capture the feedback loop: promos create discount-seekers

    if state == 0:  # normal
        if action == 0:  # no promo
            reward = 5.0  # good margin, steady demand
            next_state = 0 if random.random() < 0.9 else 1
        else:  # promo
            reward = 6.0  # short-term bump
            next_state = 1 if random.random() < 0.9 else 0  # customers learn to wait for discounts
    else:  # state == 1 discount-seeking
        if action == 0:  # no promo
            reward = 1.0  # they don't buy much without discount (pain now)
            next_state = 0 if random.random() < 0.6 else 1  # some "reset" over time
        else:  # promo
            reward = 3.0  # sales happen, but margin is worse
            next_state = 1 if random.random() < 0.9 else 0  # keeps them discount-trained

    return next_state, reward

# --- Q-learning (tabular) ---
Q = [[0.0, 0.0],   # Q[state0][action0/1]
     [0.0, 0.0]]   # Q[state1][action0/1]

alpha = 0.1   # learning rate
gamma = 0.95  # long-term value weight
eps = 0.2     # exploration

state = 0
for t in range(30_000):
    # epsilon-greedy action selection
    if random.random() < eps:
        action = random.choice([0, 1])
    else:
        action = 0 if Q[state][0] >= Q[state][1] else 1

    next_state, reward = step(state, action)

    # Q update
    best_next = max(Q[next_state])
    Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])

    state = next_state

# Learned policy
policy = {0: ("NO_PROMO" if Q[0][0] >= Q[0][1] else "PROMO"),
          1: ("NO_PROMO" if Q[1][0] >= Q[1][1] else "PROMO")}

print("Learned Q-values:")
print(" state 0 (normal):          no_promo =", round(Q[0][0], 2), " promo =", round(Q[0][1], 2))
print(" state 1 (discount-seeking): no_promo =", round(Q[1][0], 2), " promo =", round(Q[1][1], 2))
print("\nLearned policy:", policy)

# Quick simulation using learned policy to show long-run behavior
def run_sim(steps=2000):
    s = 0
    total = 0.0
    promo_count = 0
    for _ in range(steps):
        a = 0 if Q[s][0] >= Q[s][1] else 1
        promo_count += a
        s, r = step(s, a)
        total += r
    return total / steps, promo_count / steps

avg_profit, promo_rate = run_sim()
print("\nLong-run avg profit per step:", round(avg_profit, 2))
print("Promo rate:", round(100 * promo_rate, 1), "%")
