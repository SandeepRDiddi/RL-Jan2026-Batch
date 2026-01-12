import random

# -----------------------------
# Retail Coupon Feedback Loop
# -----------------------------
STATES = ["LOYAL", "PRICE_SENSITIVE", "COUPON_ADDICT"]
ACTIONS = ["NO_COUPON", "SEND_COUPON"]

LOYAL, PRICE_SENSITIVE, COUPON_ADDICT = 0, 1, 2
NO_COUPON, SEND_COUPON = 0, 1

# Business assumptions (simple and explainable):
# - Base margin per order = 100
# - Coupon reduces margin (e.g., 20% off) -> margin drops
BASE_MARGIN = 100
COUPON_MARGIN = 60  # profit per order when coupon is used

# Demand (orders per day) depends on state and action
# These are "expected" orders; reward = orders * margin
DEMAND = {
    LOYAL:          {NO_COUPON: 1.0, SEND_COUPON: 1.2},
    PRICE_SENSITIVE:{NO_COUPON: 0.7, SEND_COUPON: 1.3},
    COUPON_ADDICT:  {NO_COUPON: 0.2, SEND_COUPON: 1.1},
}

def step(state, action):
    """
    Returns (next_state, reward) for one day.

    Feedback loop is in the transition probabilities:
    - Sending coupons moves people towards COUPON_ADDICT
    - Not sending coupons can slowly recover them back to LOYAL
    """
    # reward = expected orders * expected margin
    orders = DEMAND[state][action]
    margin = BASE_MARGIN if action == NO_COUPON else COUPON_MARGIN
    reward = orders * margin

    # Transition dynamics (the learning/training effect)
    r = random.random()

    if state == LOYAL:
        if action == SEND_COUPON:
            # You might "teach" loyal people to expect coupons
            next_state = PRICE_SENSITIVE if r < 0.70 else LOYAL
        else:
            next_state = LOYAL if r < 0.90 else PRICE_SENSITIVE

    elif state == PRICE_SENSITIVE:
        if action == SEND_COUPON:
            # Repeated coupons push them into coupon addiction
            next_state = COUPON_ADDICT if r < 0.65 else PRICE_SENSITIVE
        else:
            # No coupon can slowly rebuild normal behavior
            next_state = LOYAL if r < 0.35 else PRICE_SENSITIVE

    else:  # COUPON_ADDICT
        if action == SEND_COUPON:
            # Coupon addicts stay addicted if you keep feeding coupons
            next_state = COUPON_ADDICT if r < 0.90 else PRICE_SENSITIVE
        else:
            # Withholding coupons can recover, but slowly
            next_state = PRICE_SENSITIVE if r < 0.55 else COUPON_ADDICT

    return next_state, reward

# -----------------------------
# Q-Learning
# -----------------------------
Q = [[0.0 for _ in ACTIONS] for _ in STATES]

alpha = 0.10   # learning rate
gamma = 0.95   # long-term importance
eps = 0.20     # exploration

state = LOYAL
for t in range(50_000):
    # epsilon-greedy
    if random.random() < eps:
        action = random.choice([NO_COUPON, SEND_COUPON])
    else:
        action = NO_COUPON if Q[state][NO_COUPON] >= Q[state][SEND_COUPON] else SEND_COUPON

    next_state, reward = step(state, action)

    # Bellman update
    Q[state][action] += alpha * (reward + gamma * max(Q[next_state]) - Q[state][action])
    state = next_state

# Learned policy
policy = {}
for s in range(len(STATES)):
    best_a = NO_COUPON if Q[s][NO_COUPON] >= Q[s][SEND_COUPON] else SEND_COUPON
    policy[STATES[s]] = ACTIONS[best_a]

print("Learned Q-values (higher = better long-run profit):")
for s in range(len(STATES)):
    print(f" {STATES[s]:14s}  NO_COUPON={Q[s][NO_COUPON]:8.2f}   SEND_COUPON={Q[s][SEND_COUPON]:8.2f}")

print("\nLearned Policy (what RL recommends):")
for k, v in policy.items():
    print(f"  {k:14s} -> {v}")

# -----------------------------
# Compare against baseline policies
# -----------------------------
def run_policy(policy_fn, days=5000):
    s = LOYAL
    total = 0.0
    coupon_days = 0
    visits = [0, 0, 0]
    for _ in range(days):
        visits[s] += 1
        a = policy_fn(s)
        coupon_days += (a == SEND_COUPON)
        s, r = step(s, a)
        total += r
    return total / days, coupon_days / days, [v / days for v in visits]

def always_coupon(_s): return SEND_COUPON
def never_coupon(_s):  return NO_COUPON
def rl_policy(s):
    return NO_COUPON if Q[s][NO_COUPON] >= Q[s][SEND_COUPON] else SEND_COUPON

avg_rl, coup_rl, dist_rl = run_policy(rl_policy)
avg_al, coup_al, dist_al = run_policy(always_coupon)
avg_nv, coup_nv, dist_nv = run_policy(never_coupon)

print("\n--- Policy Comparison (simulation) ---")
print(f"RL Policy      : avg_profit/day={avg_rl:6.2f}, coupon_rate={100*coup_rl:5.1f}% , state_mix={dict(zip(STATES, dist_rl))}")
print(f"Always Coupon  : avg_profit/day={avg_al:6.2f}, coupon_rate={100*coup_al:5.1f}% , state_mix={dict(zip(STATES, dist_al))}")
print(f"Never Coupon   : avg_profit/day={avg_nv:6.2f}, coupon_rate={100*coup_nv:5.1f}% , state_mix={dict(zip(STATES, dist_nv))}")
