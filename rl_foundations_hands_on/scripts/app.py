import random
import streamlit as st
import matplotlib.pyplot as plt

STATES = ["LOYAL", "PRICE_SENSITIVE", "COUPON_ADDICT"]
ACTIONS = ["NO_COUPON", "SEND_COUPON"]
LOYAL, PRICE_SENSITIVE, COUPON_ADDICT = 0, 1, 2
NO_COUPON, SEND_COUPON = 0, 1

def make_env(base_margin, coupon_margin):
    DEMAND = {
        LOYAL:          {NO_COUPON: 1.0, SEND_COUPON: 1.2},
        PRICE_SENSITIVE:{NO_COUPON: 0.7, SEND_COUPON: 1.3},
        COUPON_ADDICT:  {NO_COUPON: 0.2, SEND_COUPON: 1.1},
    }

    def step(state, action):
        orders = DEMAND[state][action]
        margin = base_margin if action == NO_COUPON else coupon_margin
        reward = orders * margin

        r = random.random()
        if state == LOYAL:
            if action == SEND_COUPON:
                next_state = PRICE_SENSITIVE if r < 0.70 else LOYAL
            else:
                next_state = LOYAL if r < 0.90 else PRICE_SENSITIVE

        elif state == PRICE_SENSITIVE:
            if action == SEND_COUPON:
                next_state = COUPON_ADDICT if r < 0.65 else PRICE_SENSITIVE
            else:
                next_state = LOYAL if r < 0.35 else PRICE_SENSITIVE

        else:  # COUPON_ADDICT
            if action == SEND_COUPON:
                next_state = COUPON_ADDICT if r < 0.90 else PRICE_SENSITIVE
            else:
                next_state = PRICE_SENSITIVE if r < 0.55 else COUPON_ADDICT

        return next_state, reward, orders, margin

    return step

def train_q_learning(step_fn, episodes, steps_per_ep, alpha, gamma, eps):
    Q = [[0.0, 0.0] for _ in STATES]
    for _ in range(episodes):
        s = LOYAL
        for _ in range(steps_per_ep):
            if random.random() < eps:
                a = random.choice([0, 1])
            else:
                a = 0 if Q[s][0] >= Q[s][1] else 1

            ns, r, *_ = step_fn(s, a)
            Q[s][a] += alpha * (r + gamma * max(Q[ns]) - Q[s][a])
            s = ns
    return Q

def simulate(step_fn, policy_fn, days, seed):
    random.seed(seed)
    s = LOYAL
    states = []
    actions = []
    rewards = []

    for _ in range(days):
        a = policy_fn(s)
        ns, r, *_ = step_fn(s, a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = ns

    return states, actions, rewards

st.title("Retail RL Demo: Coupon Feedback Loop")

st.sidebar.header("Business knobs")
base_margin = st.sidebar.slider("Profit per order (no coupon)", 50, 200, 100, 5)
coupon_margin = st.sidebar.slider("Profit per order (with coupon)", 10, 150, 60, 5)

st.sidebar.header("RL knobs")
episodes = st.sidebar.slider("Training episodes", 1000, 100000, 30000, 1000)
alpha = st.sidebar.slider("alpha (learning rate)", 0.01, 0.5, 0.10, 0.01)
gamma = st.sidebar.slider("gamma (long-term weight)", 0.50, 0.99, 0.95, 0.01)
eps = st.sidebar.slider("epsilon (exploration)", 0.0, 0.5, 0.20, 0.01)

st.sidebar.header("Simulation")
days = st.sidebar.slider("Days to simulate", 30, 365, 120, 10)
seed = st.sidebar.number_input("Random seed", value=7, step=1)

step_fn = make_env(base_margin, coupon_margin)

policy_choice = st.selectbox(
    "Choose a policy to simulate",
    ["RL (Q-learning)", "Always Coupon", "Never Coupon"]
)

Q = None
if policy_choice == "RL (Q-learning)" or st.button("Train RL now"):
    Q = train_q_learning(step_fn, episodes, steps_per_ep=50, alpha=alpha, gamma=gamma, eps=eps)

def policy_fn(s):
    if policy_choice == "Always Coupon":
        return SEND_COUPON
    if policy_choice == "Never Coupon":
        return NO_COUPON
    # RL
    return 0 if Q[s][0] >= Q[s][1] else 1

if policy_choice == "RL (Q-learning)" and Q is None:
    st.info("Click 'Train RL now' to train.")
else:
    states, actions, rewards = simulate(step_fn, policy_fn, days=days, seed=seed)

    st.subheader("Key metrics")
    st.write(f"Average profit/day: **{sum(rewards)/len(rewards):.2f}**")
    st.write(f"Coupon rate: **{100 * (sum(actions)/len(actions)):.1f}%**")

    st.subheader("Learned Q-values (if RL policy)")
    if Q is not None:
        for i, sname in enumerate(STATES):
            st.write(f"{sname}: NO_COUPON={Q[i][0]:.2f}, SEND_COUPON={Q[i][1]:.2f}")

    st.subheader("Timeline view")
    fig1 = plt.figure()
    plt.plot(states)
    plt.yticks([0, 1, 2], STATES)
    plt.xlabel("Day")
    plt.ylabel("Customer state")
    st.pyplot(fig1)

    fig2 = plt.figure()
    plt.plot(rewards)
    plt.xlabel("Day")
    plt.ylabel("Profit")
    st.pyplot(fig2)

    st.subheader("Day-by-day table (first 30 days)")
    rows = []
    for i in range(min(30, len(states))):
        rows.append({
            "Day": i + 1,
            "State": STATES[states[i]],
            "Action": ACTIONS[actions[i]],
            "Profit": round(rewards[i], 2),
        })
    st.dataframe(rows, use_container_width=True)
