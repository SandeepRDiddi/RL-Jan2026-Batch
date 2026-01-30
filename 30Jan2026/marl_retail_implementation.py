"""
================================================================================
MULTI-AGENT REINFORCEMENT LEARNING (MARL) FOR RETAIL DOMAIN
================================================================================
Complete Implementation of: MADDPG, MAPPO, and QMIX

This implementation aligns with the "Multi-Agent RL Fundamentals: CTDE & Algorithms"
training module, demonstrating all concepts in a retail context.

Author: AI Implementation for Educational Purposes
Domain: Retail (Dynamic Pricing, Inventory Management, Order Fulfillment)
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

print("=" * 80)
print("MULTI-AGENT REINFORCEMENT LEARNING FOR RETAIL")
print("Implementing: MADDPG, MAPPO, and QMIX")
print("=" * 80)
print()

# =============================================================================
# SECTION 1: RETAIL ENVIRONMENT SIMULATIONS
# =============================================================================
# Aligned with Document Section: "From Single-Agent to Multi-Agent RL"
#
# Key Concepts from Document:
# - Stochastic Games: N agents, joint action space, transition dynamics depend on ALL
# - Non-stationarity: Other agents change their policies constantly
# - Credit assignment: When multiple agents act, who caused the outcome?
# =============================================================================

print("\n" + "="*80)
print("SECTION 1: RETAIL ENVIRONMENTS")
print("="*80)

class RetailPricingEnvironment:
    """
    Multi-Agent Dynamic Pricing Environment

    Document Reference: "3-Store Pricing Game" Case Study

    Setup (from document):
    - Three stores, each sets price p_i ∈ [1, 10]
    - Demand: D_i = 100 - 2*p_i + 0.5*(p_j + p_k)  [substitution effect]
    - Profit: r_i = p_i * D_i

    This demonstrates:
    - Non-stationary environment e(other agents' prices affect demand)
    - Conflicting incentives (one store's profit may hurt others)
    - Nash Equilibrium concept (stable pricing strategies)
    """

    def __init__(self, n_stores: int = 3, min_price: float = 1.0, max_price: float = 10.0):
        self.n_agents = n_stores
        self.min_price = min_price
        self.max_price = max_price

        # State includes: recent demand history, current inventory levels
        self.obs_dim = 4  # [last_demand, inventory, competitor_avg_price, time_of_day]
        self.action_dim = 1  # Continuous price setting

        # Episode tracking
        self.current_step = 0
        self.max_steps = 50

        # History for observations
        self.demand_history = [deque(maxlen=5) for _ in range(n_stores)]
        self.price_history = [deque(maxlen=5) for _ in range(n_stores)]

        print(f"  ✓ Pricing Environment: {n_stores} stores, prices in [{min_price}, {max_price}]")

    def reset(self) -> List[np.ndarray]:
        """Reset environment and return initial observations for each agent."""
        self.current_step = 0

        # Initialize with random demand history
        for i in range(self.n_agents):
            self.demand_history[i].clear()
            self.price_history[i].clear()
            for _ in range(5):
                self.demand_history[i].append(np.random.uniform(40, 60))
                self.price_history[i].append(np.random.uniform(4, 6))

        # Return observations for each agent
        observations = []
        for i in range(self.n_agents):
            obs = self._get_observation(i)
            observations.append(obs)
        return observations

    def _get_observation(self, agent_id: int) -> np.ndarray:
        """
        Each agent's observation (what they can see locally).

        Document Reference: "CTDE - Actor πᵢ only depends on oᵢ"

        Agents can only see:
        - Their own recent demand
        - Their own inventory level
        - Average competitor price (partial info)
        - Time of day (demand varies)
        """
        avg_demand = np.mean(list(self.demand_history[agent_id])) if self.demand_history[agent_id] else 50
        inventory = np.random.uniform(0.3, 0.8)  # Simulated inventory level

        # Competitors' average price (imperfect information)
        other_prices = [np.mean(list(self.price_history[j]))
                       for j in range(self.n_agents) if j != agent_id]
        competitor_avg = np.mean(other_prices) if other_prices else 5.0

        time_of_day = (self.current_step % 24) / 24.0  # Normalized

        return np.array([avg_demand / 100.0, inventory, competitor_avg / 10.0, time_of_day],
                       dtype=np.float32)

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], bool, dict]:
        """
        Execute joint action and return next states, rewards.

        Document Reference: "Stochastic Games"
        - Transition dynamics: P(s'|s, a₁, ..., aₙ)
        - Reward for agent i: Rᵢ(s, a₁, ..., aₙ) depends on ALL actions
        """
        # Convert actions to prices
        prices = []
        for a in actions:
            price = float(a[0]) if isinstance(a, np.ndarray) else float(a)
            price = np.clip(price, self.min_price, self.max_price)
            prices.append(price)

        # Calculate demand for each store (depends on ALL prices!)
        # D_i = 100 - 2*p_i + 0.5*sum(p_j for j≠i)
        demands = []
        rewards = []

        for i in range(self.n_agents):
            other_prices_sum = sum(prices[j] for j in range(self.n_agents) if j != i)

            # Demand equation from document
            demand = 100 - 2 * prices[i] + 0.5 * other_prices_sum
            demand = max(0, demand + np.random.normal(0, 5))  # Add noise

            # Profit = price × demand
            profit = prices[i] * demand

            demands.append(demand)
            rewards.append(profit / 100.0)  # Normalized reward

            # Update history
            self.demand_history[i].append(demand)
            self.price_history[i].append(prices[i])

        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Get next observations
        next_observations = [self._get_observation(i) for i in range(self.n_agents)]

        info = {'prices': prices, 'demands': demands, 'raw_profits': [r * 100 for r in rewards]}

        return next_observations, rewards, done, info

    def get_state(self) -> np.ndarray:
        """
        Get full state (for centralized training).

        Document Reference: "CTDE - Full state: s"
        The centralized critic has access to complete information.
        """
        state = []
        for i in range(self.n_agents):
            state.extend(list(self.demand_history[i])[-3:])  # Last 3 demands
            state.extend(list(self.price_history[i])[-3:])   # Last 3 prices
        state.append(self.current_step / self.max_steps)
        return np.array(state, dtype=np.float32)


class RetailInventoryEnvironment:
    """
    Multi-Agent Inventory Management Environment

    Document Reference: "3-Warehouse Restocking Network" Case Study

    Setup (from document):
    - 3 warehouses, limited stock, can redistribute (expensive)
    - Daily stochastic demand
    - Action: quantity to restock locally

    This is a COOPERATIVE environment where all agents want inventory efficiency.
    Perfect for MAPPO due to high stochasticity and need for stability.
    """

    def __init__(self, n_warehouses: int = 3, max_capacity: int = 100):
        self.n_agents = n_warehouses
        self.max_capacity = max_capacity

        self.obs_dim = 5  # [inventory, incoming_shipment, predicted_demand, storage_cost, day_of_week]
        self.action_dim = 1  # Continuous: restock quantity (normalized)

        self.current_step = 0
        self.max_steps = 30  # One month

        # Initialize inventories
        self.inventories = np.zeros(n_warehouses)
        self.pending_shipments = np.zeros(n_warehouses)

        print(f"  ✓ Inventory Environment: {n_warehouses} warehouses, capacity {max_capacity}")

    def reset(self) -> List[np.ndarray]:
        """Reset to initial inventory levels."""
        self.current_step = 0
        self.inventories = np.random.uniform(30, 70, self.n_agents)
        self.pending_shipments = np.zeros(self.n_agents)

        return [self._get_observation(i) for i in range(self.n_agents)]

    def _get_observation(self, agent_id: int) -> np.ndarray:
        """Local observation for each warehouse."""
        inventory = self.inventories[agent_id] / self.max_capacity
        incoming = self.pending_shipments[agent_id] / self.max_capacity

        # Predicted demand (with uncertainty)
        base_demand = 20 + 10 * np.sin(2 * np.pi * self.current_step / 7)  # Weekly pattern
        predicted_demand = base_demand / self.max_capacity

        storage_cost = 0.1 * inventory  # Higher inventory = higher cost
        day_of_week = (self.current_step % 7) / 7.0

        return np.array([inventory, incoming, predicted_demand, storage_cost, day_of_week],
                       dtype=np.float32)

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], bool, dict]:
        """
        Execute restocking decisions.

        Cooperative reward: All agents share the same reward (total efficiency).
        This is why QMIX's monotonicity assumption holds!
        """
        # Process actions (restock quantities)
        restock_quantities = []
        for i, a in enumerate(actions):
            qty = float(a[0]) if isinstance(a, np.ndarray) else float(a)
            qty = np.clip(qty * 50, 0, 50)  # Denormalize to 0-50 units
            restock_quantities.append(qty)

        # Generate actual demand (stochastic)
        base_demand = 20 + 10 * np.sin(2 * np.pi * self.current_step / 7)
        demands = np.random.normal(base_demand, 5, self.n_agents)
        demands = np.maximum(0, demands)

        # Update inventories
        total_holding_cost = 0
        total_stockout_cost = 0
        total_order_cost = 0

        for i in range(self.n_agents):
            # Add incoming shipments
            self.inventories[i] += self.pending_shipments[i]
            self.pending_shipments[i] = restock_quantities[i]  # Arrives next step

            # Fulfill demand
            fulfilled = min(self.inventories[i], demands[i])
            stockout = demands[i] - fulfilled
            self.inventories[i] -= fulfilled

            # Cap at capacity
            self.inventories[i] = min(self.inventories[i], self.max_capacity)

            # Costs
            total_holding_cost += 0.1 * self.inventories[i]
            total_stockout_cost += 2.0 * stockout  # Stockouts are expensive!
            total_order_cost += 0.5 * restock_quantities[i]

        # COOPERATIVE REWARD: Same for all agents
        # Document: "Fully cooperative: All agents share same reward"
        total_cost = total_holding_cost + total_stockout_cost + total_order_cost
        shared_reward = -total_cost / (self.n_agents * 10)  # Normalized negative cost

        rewards = [shared_reward] * self.n_agents

        self.current_step += 1
        done = self.current_step >= self.max_steps

        next_obs = [self._get_observation(i) for i in range(self.n_agents)]

        info = {
            'inventories': self.inventories.copy(),
            'demands': demands,
            'holding_cost': total_holding_cost,
            'stockout_cost': total_stockout_cost
        }

        return next_obs, rewards, done, info

    def get_state(self) -> np.ndarray:
        """Full state for centralized training."""
        state = list(self.inventories / self.max_capacity)
        state.extend(list(self.pending_shipments / self.max_capacity))
        state.append(self.current_step / self.max_steps)
        return np.array(state, dtype=np.float32)


class RetailFulfillmentEnvironment:
    """
    Multi-Agent Order Fulfillment Environment

    Document Reference: "Click-and-Collect: 10-Store Network" Case Study

    Setup (from document):
    - N stores, online order arrives
    - ONE store must fulfill (constraint)
    - Cost: c_i = distance + inventory penalty

    This is DISCRETE action space (which store fulfills) and fully COOPERATIVE.
    Perfect for QMIX with value factorization!
    """

    def __init__(self, n_stores: int = 5):
        self.n_agents = n_stores

        # Store locations (for distance calculation)
        self.store_locations = np.random.rand(n_stores, 2) * 100  # 100x100 grid

        self.obs_dim = 4  # [inventory_level, distance_to_customer, fulfillment_capacity, time_pressure]
        self.n_actions = 2  # DISCRETE: 0 = don't fulfill, 1 = fulfill

        self.current_step = 0
        self.max_steps = 20

        # Current order
        self.customer_location = None
        self.inventories = None

        print(f"  ✓ Fulfillment Environment: {n_stores} stores, discrete actions")

    def reset(self) -> List[np.ndarray]:
        """Reset and generate new customer order."""
        self.current_step = 0
        self.inventories = np.random.uniform(10, 50, self.n_agents)
        self._generate_new_order()

        return [self._get_observation(i) for i in range(self.n_agents)]

    def _generate_new_order(self):
        """Generate random customer location."""
        self.customer_location = np.random.rand(2) * 100

    def _get_observation(self, agent_id: int) -> np.ndarray:
        """Local observation for each store."""
        inventory = self.inventories[agent_id] / 50.0

        # Distance to customer
        dist = np.linalg.norm(self.store_locations[agent_id] - self.customer_location)
        dist_normalized = dist / 141.4  # Max distance in 100x100 grid

        # Fulfillment capacity (based on current load)
        capacity = np.random.uniform(0.3, 1.0)

        # Time pressure (urgency)
        time_pressure = np.random.uniform(0, 1)

        return np.array([inventory, dist_normalized, capacity, time_pressure],
                       dtype=np.float32)

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], bool, dict]:
        """
        Execute fulfillment decisions.

        Document Reference: "QMIX - Monotonicity ensures greedy execution is optimal"

        Each store independently decides, but only ONE should fulfill.
        The mixing network combines Q-values to select optimal store.
        """
        # Find which stores want to fulfill
        fulfilling_stores = [i for i, a in enumerate(actions) if a == 1]

        if len(fulfilling_stores) == 0:
            # No one wants to fulfill - bad! High penalty
            shared_reward = -5.0
            fulfilling_store = -1
        elif len(fulfilling_stores) == 1:
            # Perfect - exactly one store fulfills
            fulfilling_store = fulfilling_stores[0]

            # Calculate cost
            dist = np.linalg.norm(self.store_locations[fulfilling_store] - self.customer_location)
            inventory_penalty = max(0, 20 - self.inventories[fulfilling_store]) * 0.1

            cost = dist / 100.0 + inventory_penalty
            shared_reward = 1.0 - cost  # Positive reward for fulfillment minus costs

            # Update inventory
            self.inventories[fulfilling_store] -= 1
        else:
            # Multiple stores trying to fulfill - coordination failure
            # Pick the best one but add penalty for confusion
            costs = []
            for store in fulfilling_stores:
                dist = np.linalg.norm(self.store_locations[store] - self.customer_location)
                inv_penalty = max(0, 20 - self.inventories[store]) * 0.1
                costs.append(dist / 100.0 + inv_penalty)

            best_idx = np.argmin(costs)
            fulfilling_store = fulfilling_stores[best_idx]

            shared_reward = 0.5 - costs[best_idx] - 0.3 * (len(fulfilling_stores) - 1)  # Coordination penalty
            self.inventories[fulfilling_store] -= 1

        # COOPERATIVE: Same reward for all
        rewards = [shared_reward] * self.n_agents

        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Generate new order for next step
        self._generate_new_order()

        next_obs = [self._get_observation(i) for i in range(self.n_agents)]

        info = {'fulfilling_store': fulfilling_store, 'n_trying': len(fulfilling_stores)}

        return next_obs, rewards, done, info

    def get_state(self) -> np.ndarray:
        """Full state for centralized training."""
        if self.inventories is None:
            self.reset()
        state = list(self.inventories / 50.0)
        state.extend(list(self.customer_location / 100.0))
        state.append(self.current_step / self.max_steps)
        return np.array(state, dtype=np.float32)


# Initialize environments
print("\nInitializing Retail Environments:")
pricing_env = RetailPricingEnvironment(n_stores=3)
inventory_env = RetailInventoryEnvironment(n_warehouses=3)
fulfillment_env = RetailFulfillmentEnvironment(n_stores=5)


# =============================================================================
# SECTION 2: NEURAL NETWORK ARCHITECTURES
# =============================================================================
# Aligned with Document Section: "CTDE Architecture"
#
# Key Concepts:
# - Actor networks: π_i(a_i|o_i) - take local observation, output action
# - Critic networks: Q^cen(s, a_1, ..., a_n) - take full state and all actions
# - The actor only depends on o_i, but gets gradients from joint evaluation
# =============================================================================

print("\n" + "="*80)
print("SECTION 2: NEURAL NETWORK ARCHITECTURES")
print("="*80)


class Actor(nn.Module):
    """
    Actor Network for MADDPG and MAPPO

    Document Reference: "Actor πᵢ only depends on oᵢ"

    Takes local observation o_i and outputs:
    - MADDPG: Deterministic action μ_i(o_i)
    - MAPPO: Stochastic policy π_i(a_i|o_i) (mean and std)

    At execution time, this runs INDEPENDENTLY - no communication!
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64,
                 deterministic: bool = True):
        super(Actor, self).__init__()

        self.deterministic = deterministic

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        if deterministic:
            # MADDPG: Output deterministic action
            self.fc3 = nn.Linear(hidden_dim, action_dim)
        else:
            # MAPPO: Output mean and log_std for Gaussian policy
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        if self.deterministic:
            # Document: "a = μ(o)" - deterministic policy
            return torch.tanh(self.fc3(x))  # Bounded action
        else:
            # Document: "πᵢ(aᵢ|oᵢ)" - stochastic policy
            mean = self.mean(x)
            log_std = torch.clamp(self.log_std(x), -2, 2)
            return mean, log_std

    def sample_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from stochastic policy (MAPPO)."""
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)

        # Reparameterization trick
        noise = torch.randn_like(mean)
        action = torch.tanh(mean + std * noise)

        # Log probability (for PPO ratio calculation)
        log_prob = -0.5 * ((action - mean) / std).pow(2) - log_std - 0.5 * np.log(2 * np.pi)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class CentralizedCritic(nn.Module):
    """
    Centralized Critic for MADDPG and MAPPO

    Document Reference: "Joint Value Function"
    Q^cen(s, a₁, ..., aₙ) = 𝔼[Σₜ γᵗ rₜ | s, a₁, ..., aₙ]

    CRITICAL: This has access to FULL state and ALL actions during TRAINING.
    It learns to evaluate the joint action - enabling credit assignment.

    "Critic sees everything during training and identifies which agent's
    action helps or hurts."
    """

    def __init__(self, state_dim: int, all_actions_dim: int, hidden_dim: int = 128):
        super(CentralizedCritic, self).__init__()

        # Input: full state + all agents' actions concatenated
        input_dim = state_dim + all_actions_dim
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Single Q-value output

    def forward(self, state: torch.Tensor, all_actions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate joint action given full state.

        Document: "Uses full joint information"
        y_i = r_i + γ Q_i^-(s', μ₁^-(o₁'), ..., μₙ^-(oₙ'))
        """
        x = torch.cat([state, all_actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ValueNetwork(nn.Module):
    """
    Centralized Value Network for MAPPO

    Document Reference: "V^cen(s) = value function"

    Used for advantage estimation: Â_i = return - V^cen(s)
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(ValueNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QNetwork(nn.Module):
    """
    Individual Q-Network for QMIX

    Document Reference: "Q-Networks: oᵢ → Q_netᵢ → Qᵢ(aᵢ)"

    Each agent has its own Q-network that takes local observation
    and outputs Q-values for each discrete action.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Output Q-values for all actions."""
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QMIXMixingNetwork(nn.Module):
    """
    QMIX Mixing Network with Monotonicity Constraint

    Document Reference: "QMIX: Mixing Network"

    CRITICAL CONSTRAINT: ∂Q_total / ∂Q_i ≥ 0 for all agents

    "Enforce monotonicity via architecture:
    w1 = torch.abs(self.w1.weight)  # abs() ensures w ≥ 0"

    This ensures: Each agent independently maximizing its Q_i
    AUTOMATICALLY maximizes Q_total!
    """

    def __init__(self, n_agents: int, state_dim: int, mixing_embed_dim: int = 32):
        super(QMIXMixingNetwork, self).__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim

        # Hypernetworks: generate weights from state
        # These produce the mixing network weights conditioned on state

        # First layer weights (n_agents → embed_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_agents * mixing_embed_dim)
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)

        # Second layer weights (embed_dim → 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, mixing_embed_dim)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Mix individual Q-values into Q_total.

        Document: "Q_total = mixing_net(Q₁, Q₂, ..., Qₙ | s)"

        The state conditions the mixing, but monotonicity is maintained
        via absolute value on weights.
        """
        batch_size = q_values.shape[0]

        # Generate weights from state (hypernetwork)
        w1 = self.hyper_w1(state).view(batch_size, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)

        w2 = self.hyper_w2(state).view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        # CRITICAL: Absolute value ensures monotonicity!
        # Document: "torch.abs() on weights ensures monotonicity through each layer"
        w1 = torch.abs(w1)
        w2 = torch.abs(w2)

        # Forward pass through mixing network
        # q_values: (batch, n_agents) → (batch, 1, n_agents)
        q_values = q_values.unsqueeze(1)

        # First layer
        hidden = F.elu(torch.bmm(q_values, w1) + b1)

        # Second layer
        q_total = torch.bmm(hidden, w2) + b2

        return q_total.squeeze(-1).squeeze(-1)

print("  ✓ Actor Network (for local decision making)")
print("  ✓ Centralized Critic (for joint value estimation)")
print("  ✓ Value Network (for MAPPO advantage estimation)")
print("  ✓ Q-Network (individual Q-values for QMIX)")
print("  ✓ QMIX Mixing Network (with monotonicity constraint)")


# =============================================================================
# SECTION 3: MADDPG IMPLEMENTATION
# =============================================================================
# Document Reference: "MADDPG: Multi-Agent Deep Deterministic Policy Gradient"
#
# Key Points from Document:
# - Deterministic policy: a = μ(o)
# - Off-policy learning with replay buffer
# - Centralized critic, decentralized actors
# - Gradient: ∇_θᵢ Jᵢ = 𝔼[∇_θᵢ μᵢ(oᵢ) · ∇_aᵢ Qᵢ^cen(s, a₁, ..., aₙ)]
# =============================================================================

print("\n" + "="*80)
print("SECTION 3: MADDPG IMPLEMENTATION")
print("="*80)

# Experience tuple for replay buffer
Experience = namedtuple('Experience',
                        ['observations', 'state', 'actions', 'rewards',
                         'next_observations', 'next_state', 'done'])

class ReplayBuffer:
    """
    Experience Replay Buffer for Off-Policy Learning

    Document Reference: "Store in replay buffer"

    MADDPG uses off-policy learning - it can learn from past experiences.
    This is more sample-efficient than on-policy methods like MAPPO.
    """

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class MADDPG:
    """
    Multi-Agent Deep Deterministic Policy Gradient

    Document Reference: Full MADDPG section

    "MADDPG Extension: Use centralized critics with multi-agent"

    Key Algorithm Steps (from document):
    1. Collection: All agents act in parallel with exploration noise
    2. Critic Update: Uses full joint information
    3. Actor Update: Each agent uses own gradient
    4. Target Networks: Soft updates for stability
    """

    def __init__(self, n_agents: int, obs_dims: List[int], action_dims: List[int],
                 state_dim: int, lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.95, tau: float = 0.01):

        self.n_agents = n_agents
        self.gamma = gamma  # Discount factor
        self.tau = tau      # Soft update parameter (Document: τ ≈ 0.001)

        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.state_dim = state_dim

        # Create actor networks (one per agent)
        # Document: "Actor πᵢ only depends on oᵢ"
        self.actors = [Actor(obs_dims[i], action_dims[i], deterministic=True)
                      for i in range(n_agents)]
        self.actors_target = [Actor(obs_dims[i], action_dims[i], deterministic=True)
                             for i in range(n_agents)]

        # Create centralized critics (one per agent, but each sees EVERYTHING)
        # Document: "Critic Update: Uses full joint information"
        # Input: all observations concatenated + all actions concatenated
        total_obs_dim = sum(obs_dims)
        total_action_dim = sum(action_dims)
        self.critics = [CentralizedCritic(total_obs_dim, total_action_dim)
                       for _ in range(n_agents)]
        self.critics_target = [CentralizedCritic(total_obs_dim, total_action_dim)
                              for _ in range(n_agents)]

        # Initialize target networks
        for i in range(n_agents):
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())

        # Optimizers
        self.actor_optimizers = [optim.Adam(self.actors[i].parameters(), lr=lr_actor)
                                for i in range(n_agents)]
        self.critic_optimizers = [optim.Adam(self.critics[i].parameters(), lr=lr_critic)
                                 for i in range(n_agents)]

        # Replay buffer
        self.replay_buffer = ReplayBuffer()

        # Exploration noise (Document: "aᵢ_executed = μᵢ(oᵢ) + ε, where ε ~ N(0, σ²)")
        self.noise_scale = 0.1

        print(f"  ✓ MADDPG initialized: {n_agents} agents")
        print(f"    - Actors: Deterministic policies μᵢ(oᵢ)")
        print(f"    - Critics: Centralized Q^cen(s, a₁, ..., aₙ)")
        print(f"    - Target networks with τ = {tau}")

    def select_action(self, observations: List[np.ndarray],
                     explore: bool = True) -> List[np.ndarray]:
        """
        Select actions for all agents.

        Document: "For each i: aᵢ = μᵢ(oᵢ) + noise"

        During training, we add Gaussian noise for exploration.
        At execution, we use the deterministic policy directly.
        """
        actions = []
        for i, obs in enumerate(observations):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            with torch.no_grad():
                action = self.actors[i](obs_tensor).squeeze(0).numpy()

            if explore:
                # Add exploration noise
                noise = np.random.normal(0, self.noise_scale, action.shape)
                action = np.clip(action + noise, -1, 1)

            actions.append(action)

        return actions

    def store_experience(self, observations, state, actions, rewards,
                        next_observations, next_state, done):
        """Store transition in replay buffer."""
        exp = Experience(observations, state, actions, rewards,
                        next_observations, next_state, done)
        self.replay_buffer.push(exp)

    def update(self, batch_size: int = 64) -> Dict[str, float]:
        """
        Update all agents' networks.

        Document Reference: "MADDPG Training Loop"

        Steps:
        1. Sample batch from replay buffer
        2. Critic Update: y_i = r_i + γ Q_i^-(s', μ₁^-(o₁'), ..., μₙ^-(oₙ'))
        3. Actor Update: ∇_θᵢ μᵢ(oᵢ) · ∇_aᵢ Qᵢ(s, a₁, ..., aₙ)
        4. Soft update targets
        """
        if len(self.replay_buffer) < batch_size:
            return {}

        # Sample batch
        batch = self.replay_buffer.sample(batch_size)

        # Unpack batch
        obs_batch = [[exp.observations[i] for exp in batch] for i in range(self.n_agents)]
        state_batch = [exp.state for exp in batch]
        action_batch = [[exp.actions[i] for exp in batch] for i in range(self.n_agents)]
        reward_batch = [[exp.rewards[i] for exp in batch] for i in range(self.n_agents)]
        next_obs_batch = [[exp.next_observations[i] for exp in batch] for i in range(self.n_agents)]
        next_state_batch = [exp.next_state for exp in batch]
        done_batch = [exp.done for exp in batch]

        # Convert to tensors - use observations concatenated as "state" for critic
        # Centralized critic sees all observations concatenated
        all_obs_tensor = torch.cat([torch.FloatTensor(np.array(obs_batch[i]))
                                   for i in range(self.n_agents)], dim=1)
        all_next_obs_tensor = torch.cat([torch.FloatTensor(np.array(next_obs_batch[i]))
                                        for i in range(self.n_agents)], dim=1)
        done_tensor = torch.FloatTensor(done_batch).unsqueeze(1)

        losses = {'critic': [], 'actor': []}

        # Get all current and next actions
        all_actions = torch.cat([torch.FloatTensor(np.array(action_batch[i]))
                                for i in range(self.n_agents)], dim=1)

        # Get target actions from target actors
        with torch.no_grad():
            next_actions = []
            for i in range(self.n_agents):
                next_obs_tensor_i = torch.FloatTensor(np.array(next_obs_batch[i]))
                next_actions.append(self.actors_target[i](next_obs_tensor_i))
            all_next_actions = torch.cat(next_actions, dim=1)

        # Update each agent
        for i in range(self.n_agents):
            # --- CRITIC UPDATE ---
            # Document: "y_i = r_i + γ Q_i^-(s', μ₁^-(o₁'), ..., μₙ^-(oₙ'))"
            reward_tensor = torch.FloatTensor(reward_batch[i]).unsqueeze(1)

            with torch.no_grad():
                target_q = self.critics_target[i](all_next_obs_tensor, all_next_actions)
                y = reward_tensor + self.gamma * (1 - done_tensor) * target_q

            current_q = self.critics[i](all_obs_tensor, all_actions)

            # Document: "Loss: L_i = (Q_i(s, a) - y_i)²"
            critic_loss = F.mse_loss(current_q, y)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            losses['critic'].append(critic_loss.item())

            # --- ACTOR UPDATE ---
            # Document: "∇_θᵢ Jᵢ = 𝔼[∇_θᵢ μᵢ(oᵢ) · ∇_aᵢ Qᵢ(s, a₁, ..., aₙ)]"
            obs_tensor = torch.FloatTensor(np.array(obs_batch[i]))

            # Get current actions from all actors (but only update actor i)
            current_actions = []
            for j in range(self.n_agents):
                if j == i:
                    # Use current actor's output (with gradients)
                    current_actions.append(self.actors[j](obs_tensor))
                else:
                    # Detach other actors
                    other_obs = torch.FloatTensor(np.array(obs_batch[j]))
                    current_actions.append(self.actors[j](other_obs).detach())

            current_all_actions = torch.cat(current_actions, dim=1)

            # Actor loss: maximize Q (minimize -Q)
            actor_loss = -self.critics[i](all_obs_tensor, current_all_actions).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            losses['actor'].append(actor_loss.item())

        # --- SOFT UPDATE TARGET NETWORKS ---
        # Document: "θ_μᵢ^- ← τ θ_μᵢ + (1-τ) θ_μᵢ^-"
        self._soft_update_targets()

        return {
            'avg_critic_loss': np.mean(losses['critic']),
            'avg_actor_loss': np.mean(losses['actor'])
        }

    def _soft_update_targets(self):
        """
        Soft update target networks.

        Document: "τ ≈ 0.001" for stability
        """
        for i in range(self.n_agents):
            for target_param, param in zip(self.actors_target[i].parameters(),
                                          self.actors[i].parameters()):
                target_param.data.copy_(self.tau * param.data +
                                       (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.critics_target[i].parameters(),
                                          self.critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data +
                                       (1 - self.tau) * target_param.data)


# =============================================================================
# SECTION 4: MAPPO IMPLEMENTATION
# =============================================================================
# Document Reference: "MAPPO: Multi-Agent Proximal Policy Optimization"
#
# Key Points from Document:
# - Stochastic policy: π_i(a_i|o_i)
# - On-policy learning (fresh trajectories, no replay buffer)
# - PPO clipping for stability
# - Natural exploration via policy entropy
# =============================================================================

print("\n" + "="*80)
print("SECTION 4: MAPPO IMPLEMENTATION")
print("="*80)

class RolloutBuffer:
    """
    Rollout Buffer for On-Policy Learning (MAPPO)

    Document: "Collection (On-policy): Fresh trajectories"

    Unlike MADDPG's replay buffer, MAPPO collects fresh trajectories
    and discards them after each update. Less sample-efficient but more stable.
    """

    def __init__(self):
        self.observations = []
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, observations, state, actions, log_probs, rewards, done, values):
        self.observations.append(observations)
        self.states.append(state)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.dones.append(done)
        self.values.append(values)

    def clear(self):
        self.observations = []
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def get_batch(self, n_agents: int, gamma: float = 0.99,
                  gae_lambda: float = 0.95) -> Dict:
        """
        Compute returns and advantages using GAE.

        Document: "Â_i = return - V^cen(s)"
        """
        batch_size = len(self.rewards)

        # Compute returns and advantages
        returns = []
        advantages = []

        for i in range(n_agents):
            agent_returns = []
            agent_advantages = []

            # GAE computation
            gae = 0
            for t in reversed(range(batch_size)):
                if t == batch_size - 1:
                    next_value = 0
                else:
                    next_value = self.values[t + 1]

                delta = self.rewards[t][i] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
                gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
                agent_advantages.insert(0, gae)
                agent_returns.insert(0, gae + self.values[t])

            returns.append(agent_returns)
            advantages.append(agent_advantages)

        return {
            'observations': self.observations,
            'states': self.states,
            'actions': self.actions,
            'log_probs': self.log_probs,
            'returns': returns,
            'advantages': advantages
        }


class MAPPO:
    """
    Multi-Agent Proximal Policy Optimization

    Document Reference: Full MAPPO section

    "MAPPO Extension: Stochastic policy πᵢ(aᵢ|oᵢ) and on-policy learning"

    Key differences from MADDPG:
    - Stochastic policies (natural exploration)
    - On-policy (fresh data only)
    - PPO clipping (prevents large updates → stability)
    """

    def __init__(self, n_agents: int, obs_dims: List[int], action_dims: List[int],
                 state_dim: int, lr: float = 3e-4, gamma: float = 0.99,
                 clip_epsilon: float = 0.2, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, n_epochs: int = 4):

        self.n_agents = n_agents
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon  # Document: "ε" in clipping
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs  # Document: "K epochs PPO"

        # Stochastic actors (one per agent)
        # Document: "Stochastic policy πᵢ(aᵢ|oᵢ)"
        self.actors = [Actor(obs_dims[i], action_dims[i], deterministic=False)
                      for i in range(n_agents)]

        # Shared centralized value network
        # Document: "V^cen(s) = value function"
        self.value_net = ValueNetwork(state_dim)

        # Optimizers
        self.actor_optimizers = [optim.Adam(self.actors[i].parameters(), lr=lr)
                                for i in range(n_agents)]
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer()

        print(f"  ✓ MAPPO initialized: {n_agents} agents")
        print(f"    - Actors: Stochastic policies πᵢ(aᵢ|oᵢ)")
        print(f"    - Shared Value Network: V^cen(s)")
        print(f"    - PPO clip epsilon: {clip_epsilon}")
        print(f"    - K epochs: {n_epochs}")

    def select_action(self, observations: List[np.ndarray],
                     state: np.ndarray) -> Tuple[List[np.ndarray], List[float], float]:
        """
        Sample actions from stochastic policies.

        Document: "For each i: aᵢₜ ~ πᵢ(·|oᵢₜ)"

        Returns actions, log probabilities, and state value.
        """
        actions = []
        log_probs = []

        for i, obs in enumerate(observations):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action, log_prob = self.actors[i].sample_action(obs_tensor)

            actions.append(action.squeeze(0).detach().numpy())
            log_probs.append(log_prob.squeeze(0).item())

        # Get state value
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            value = self.value_net(state_tensor).item()

        return actions, log_probs, value

    def store_transition(self, observations, state, actions, log_probs,
                        rewards, done, value):
        """Store transition in rollout buffer."""
        self.rollout_buffer.add(observations, state, actions, log_probs,
                               rewards, done, value)

    def update(self) -> Dict[str, float]:
        """
        Update policy using PPO.

        Document Reference: "MAPPO Training Loop"

        Key equation:
        L^CLIP_i = 𝔼[min(rᵢₜ Âᵢₜ, clip(rᵢₜ, 1-ε, 1+ε) Âᵢₜ)]

        where r_it = π(aᵢₜ|oᵢₜ) / π_old(aᵢₜ|oᵢₜ)
        """
        # Get batch with computed returns and advantages
        batch = self.rollout_buffer.get_batch(self.n_agents, self.gamma)

        losses = {'policy': [], 'value': [], 'entropy': []}

        # PPO epochs
        # Document: "K epochs PPO"
        for _ in range(self.n_epochs):
            for i in range(self.n_agents):
                policy_loss_epoch = 0
                value_loss_epoch = 0
                entropy_loss_epoch = 0

                for t in range(len(batch['observations'])):
                    obs = torch.FloatTensor(batch['observations'][t][i]).unsqueeze(0)
                    state = torch.FloatTensor(batch['states'][t]).unsqueeze(0)
                    action = torch.FloatTensor(batch['actions'][t][i]).unsqueeze(0)
                    old_log_prob = batch['log_probs'][t][i]
                    advantage = batch['advantages'][i][t]
                    ret = batch['returns'][i][t]

                    # Get new log probability and value
                    mean, log_std = self.actors[i](obs)
                    std = torch.exp(log_std)

                    # Log probability of action under current policy
                    new_log_prob = -0.5 * ((action - mean) / std).pow(2) - log_std
                    new_log_prob = new_log_prob.sum(dim=-1)

                    # Document: "rᵢₜ = π(aᵢₜ|oᵢₜ) / π_old(aᵢₜ|oᵢₜ)"
                    ratio = torch.exp(new_log_prob - old_log_prob)

                    # Document: "L^CLIP = min(r·Â, clip(r, 1-ε, 1+ε)·Â)"
                    advantage_tensor = torch.FloatTensor([advantage])
                    surr1 = ratio * advantage_tensor
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                       1 + self.clip_epsilon) * advantage_tensor
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    # Document: "L_V = (V^cen(s) - G)²"
                    value_pred = self.value_net(state)
                    value_loss = F.mse_loss(value_pred, torch.FloatTensor([[ret]]))

                    # Entropy bonus for exploration
                    # Document: "Exploration via entropy"
                    entropy = 0.5 * (1 + np.log(2 * np.pi)) + log_std.mean()

                    # Update actor
                    self.actor_optimizers[i].zero_grad()
                    actor_loss = policy_loss - self.entropy_coef * entropy
                    actor_loss.backward()
                    self.actor_optimizers[i].step()

                    policy_loss_epoch += policy_loss.item()
                    entropy_loss_epoch += entropy.item()

                # Update value network (shared)
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

                value_loss_epoch += value_loss.item()

                losses['policy'].append(policy_loss_epoch / len(batch['observations']))
                losses['value'].append(value_loss_epoch / len(batch['observations']))
                losses['entropy'].append(entropy_loss_epoch / len(batch['observations']))

        # Clear buffer (on-policy!)
        self.rollout_buffer.clear()

        return {
            'avg_policy_loss': np.mean(losses['policy']),
            'avg_value_loss': np.mean(losses['value']),
            'avg_entropy': np.mean(losses['entropy'])
        }


# =============================================================================
# SECTION 5: QMIX IMPLEMENTATION
# =============================================================================
# Document Reference: "QMIX: Q-Learning with Value Function Factorization"
#
# Key Points from Document:
# - Value factorization: Q_total = f(Q₁, Q₂, ..., Qₙ)
# - Monotonicity constraint: ∂Q_total/∂Qᵢ ≥ 0
# - Decentralized execution: a*ᵢ = argmax Qᵢ(oᵢ, aᵢ) independently
# =============================================================================

print("\n" + "="*80)
print("SECTION 5: QMIX IMPLEMENTATION")
print("="*80)


class QMIX:
    """
    QMIX: Q-Learning with Value Function Factorization

    Document Reference: Full QMIX section

    "QMIX Factorization: Q_total = mixing_net(Q₁, Q₂, ..., Qₙ | s)"

    The key insight is MONOTONICITY:
    - If ∂Q_total/∂Qᵢ ≥ 0, then greedy execution is optimal
    - Each agent can independently choose argmax Qᵢ
    - This automatically maximizes Q_total!

    Best for: Fully cooperative, discrete actions, scalable to many agents
    """

    def __init__(self, n_agents: int, obs_dims: List[int], n_actions: int,
                 state_dim: int, lr: float = 5e-4, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995):

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma

        # Epsilon-greedy exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Individual Q-networks (one per agent)
        # Document: "Q-Networks: oᵢ → Q_netᵢ → Qᵢ(aᵢ)"
        self.q_networks = [QNetwork(obs_dims[i], n_actions) for i in range(n_agents)]
        self.q_networks_target = [QNetwork(obs_dims[i], n_actions) for i in range(n_agents)]

        # Initialize targets
        for i in range(n_agents):
            self.q_networks_target[i].load_state_dict(self.q_networks[i].state_dict())

        # Mixing network
        # Document: "QMIX Mixing Network with Monotonicity Constraint"
        self.mixing_net = QMIXMixingNetwork(n_agents, state_dim)
        self.mixing_net_target = QMIXMixingNetwork(n_agents, state_dim)
        self.mixing_net_target.load_state_dict(self.mixing_net.state_dict())

        # Collect all parameters
        params = list(self.mixing_net.parameters())
        for q_net in self.q_networks:
            params.extend(list(q_net.parameters()))

        self.optimizer = optim.Adam(params, lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer()

        print(f"  ✓ QMIX initialized: {n_agents} agents, {n_actions} actions")
        print(f"    - Individual Q-networks for each agent")
        print(f"    - Mixing network with MONOTONICITY constraint")
        print(f"    - Greedy execution is globally optimal!")

    def select_action(self, observations: List[np.ndarray],
                     explore: bool = True) -> List[int]:
        """
        Select actions using epsilon-greedy.

        Document: "a*ᵢ = argmax_aᵢ Qᵢ(oᵢ, aᵢ) independently"

        At execution, each agent independently takes greedy action.
        Monotonicity guarantees this achieves global optimum!
        """
        actions = []

        for i, obs in enumerate(observations):
            if explore and random.random() < self.epsilon:
                # Explore: random action
                action = random.randint(0, self.n_actions - 1)
            else:
                # Exploit: greedy action
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.q_networks[i](obs_tensor)
                    action = q_values.argmax(dim=1).item()

            actions.append(action)

        return actions

    def store_experience(self, observations, state, actions, rewards,
                        next_observations, next_state, done):
        """Store transition in replay buffer."""
        exp = Experience(observations, state, actions, rewards,
                        next_observations, next_state, done)
        self.replay_buffer.push(exp)

    def update(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Update Q-networks and mixing network.

        Document Reference: "QMIX Training"

        1. Get individual Q-values for each agent
        2. Mix them using mixing network (with monotonicity)
        3. Compute TD target using target networks
        4. Minimize (Q_total - y)²
        """
        if len(self.replay_buffer) < batch_size:
            return {}

        # Sample batch
        batch = self.replay_buffer.sample(batch_size)

        # Prepare batch tensors
        obs_batch = [[exp.observations[i] for exp in batch] for i in range(self.n_agents)]
        state_batch = torch.FloatTensor([exp.state for exp in batch])
        action_batch = [[exp.actions[i] for exp in batch] for i in range(self.n_agents)]
        # For QMIX, rewards should be the same (cooperative)
        reward_batch = torch.FloatTensor([exp.rewards[0] for exp in batch])  # Shared reward
        next_obs_batch = [[exp.next_observations[i] for exp in batch] for i in range(self.n_agents)]
        next_state_batch = torch.FloatTensor([exp.next_state for exp in batch])
        done_batch = torch.FloatTensor([exp.done for exp in batch])

        # Get current Q-values for taken actions
        q_values = []
        for i in range(self.n_agents):
            obs_tensor = torch.FloatTensor(np.array(obs_batch[i]))
            actions_tensor = torch.LongTensor(action_batch[i]).unsqueeze(1)

            q_all = self.q_networks[i](obs_tensor)
            q_taken = q_all.gather(1, actions_tensor).squeeze(1)
            q_values.append(q_taken)

        q_values = torch.stack(q_values, dim=1)  # (batch, n_agents)

        # Mix Q-values to get Q_total
        # Document: "Q_total = mixing_net(Q₁, Q₂, ..., Qₙ | s)"
        q_total = self.mixing_net(q_values, state_batch)

        # Compute target Q_total
        with torch.no_grad():
            next_q_values = []
            for i in range(self.n_agents):
                next_obs_tensor = torch.FloatTensor(np.array(next_obs_batch[i]))
                next_q = self.q_networks_target[i](next_obs_tensor)
                next_q_max = next_q.max(dim=1)[0]
                next_q_values.append(next_q_max)

            next_q_values = torch.stack(next_q_values, dim=1)
            next_q_total = self.mixing_net_target(next_q_values, next_state_batch)

            # TD target
            target = reward_batch + self.gamma * (1 - done_batch) * next_q_total

        # Loss and update
        loss = F.mse_loss(q_total, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mixing_net.parameters(), 10)
        for q_net in self.q_networks:
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10)
        self.optimizer.step()

        # Update targets periodically
        self._soft_update_targets(tau=0.005)

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {'loss': loss.item(), 'epsilon': self.epsilon}

    def _soft_update_targets(self, tau: float = 0.005):
        """Soft update target networks."""
        for i in range(self.n_agents):
            for target_param, param in zip(self.q_networks_target[i].parameters(),
                                          self.q_networks[i].parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.mixing_net_target.parameters(),
                                      self.mixing_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# =============================================================================
# SECTION 6: TRAINING AND DEMONSTRATION
# =============================================================================
# Run training demos for each algorithm on their suited environments
# =============================================================================

print("\n" + "="*80)
print("SECTION 6: TRAINING DEMONSTRATIONS")
print("="*80)


def train_maddpg_pricing():
    """
    Train MADDPG on Dynamic Pricing Environment

    Document Reference: "Scenario 1: Dynamic Pricing - Algorithm: MADDPG"
    Why: Continuous prices, mixed competitive setting, historical data available
    """
    print("\n" + "-"*60)
    print("TRAINING: MADDPG on Dynamic Pricing")
    print("Document: 'Continuous, historical data' → MADDPG")
    print("-"*60)

    env = RetailPricingEnvironment(n_stores=3)

    # Initialize MADDPG
    obs_dims = [env.obs_dim] * env.n_agents
    action_dims = [env.action_dim] * env.n_agents
    state_dim = len(env.get_state())

    agent = MADDPG(
        n_agents=env.n_agents,
        obs_dims=obs_dims,
        action_dims=action_dims,
        state_dim=state_dim
    )

    # Training loop
    n_episodes = 50
    episode_rewards = []

    print("\nTraining Progress:")
    for episode in range(n_episodes):
        observations = env.reset()
        state = env.get_state()
        total_rewards = np.zeros(env.n_agents)

        for step in range(env.max_steps):
            # Select actions with exploration
            actions = agent.select_action(observations, explore=True)

            # Environment step
            next_observations, rewards, done, info = env.step(actions)
            next_state = env.get_state()

            # Store experience
            agent.store_experience(
                observations, state, actions, rewards,
                next_observations, next_state, done
            )

            # Update
            if len(agent.replay_buffer) > 64:
                agent.update(batch_size=64)

            total_rewards += np.array(rewards)
            observations = next_observations
            state = next_state

            if done:
                break

        episode_rewards.append(np.mean(total_rewards))

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"  Episode {episode+1}/{n_episodes} | Avg Reward: {avg_reward:.3f}")
            if episode >= 40:
                print(f"    Sample prices: {[f'{p:.2f}' for p in info['prices']]}")

    print("\n✓ MADDPG Training Complete")
    return agent, episode_rewards


def train_mappo_inventory():
    """
    Train MAPPO on Inventory Management Environment

    Document Reference: "Scenario 2: Inventory Restocking - Algorithm: MAPPO"
    Why: Continuous, cooperative, high stochasticity. Stability important.
    """
    print("\n" + "-"*60)
    print("TRAINING: MAPPO on Inventory Management")
    print("Document: 'Cooperative, stochastic, stability needed' → MAPPO")
    print("-"*60)

    env = RetailInventoryEnvironment(n_warehouses=3)

    # Initialize MAPPO
    obs_dims = [env.obs_dim] * env.n_agents
    action_dims = [env.action_dim] * env.n_agents
    state_dim = len(env.get_state())

    agent = MAPPO(
        n_agents=env.n_agents,
        obs_dims=obs_dims,
        action_dims=action_dims,
        state_dim=state_dim
    )

    # Training loop
    n_episodes = 50
    episode_rewards = []

    print("\nTraining Progress:")
    for episode in range(n_episodes):
        observations = env.reset()
        state = env.get_state()
        total_rewards = np.zeros(env.n_agents)

        for step in range(env.max_steps):
            # Select actions (stochastic)
            actions, log_probs, value = agent.select_action(observations, state)

            # Environment step
            next_observations, rewards, done, info = env.step(actions)
            next_state = env.get_state()

            # Store transition
            agent.store_transition(
                observations, state, actions, log_probs,
                rewards, done, value
            )

            total_rewards += np.array(rewards)
            observations = next_observations
            state = next_state

            if done:
                break

        # Update after episode (on-policy)
        if len(agent.rollout_buffer.rewards) > 0:
            agent.update()

        episode_rewards.append(np.mean(total_rewards))

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"  Episode {episode+1}/{n_episodes} | Avg Reward: {avg_reward:.3f}")
            if episode >= 40:
                print(f"    Inventories: {[f'{inv:.1f}' for inv in info['inventories']]}")

    print("\n✓ MAPPO Training Complete")
    return agent, episode_rewards


def train_qmix_fulfillment():
    """
    Train QMIX on Order Fulfillment Environment

    Document Reference: "Scenario 3: Order Fulfillment - Algorithm: QMIX"
    Why: Discrete (which store), cooperative, benefits from factorization
    """
    print("\n" + "-"*60)
    print("TRAINING: QMIX on Order Fulfillment")
    print("Document: 'Discrete, cooperative, factorization' → QMIX")
    print("-"*60)

    env = RetailFulfillmentEnvironment(n_stores=5)

    # Initialize QMIX
    obs_dims = [env.obs_dim] * env.n_agents
    state_dim = len(env.get_state())

    agent = QMIX(
        n_agents=env.n_agents,
        obs_dims=obs_dims,
        n_actions=env.n_actions,
        state_dim=state_dim
    )

    # Training loop
    n_episodes = 100
    episode_rewards = []
    coordination_success = []

    print("\nTraining Progress:")
    for episode in range(n_episodes):
        observations = env.reset()
        state = env.get_state()
        total_reward = 0
        successful_coords = 0
        total_orders = 0

        for step in range(env.max_steps):
            # Select actions (epsilon-greedy)
            actions = agent.select_action(observations, explore=True)

            # Environment step
            next_observations, rewards, done, info = env.step(actions)
            next_state = env.get_state()

            # Store experience
            agent.store_experience(
                observations, state, actions, rewards,
                next_observations, next_state, done
            )

            # Update
            if len(agent.replay_buffer) > 32:
                agent.update(batch_size=32)

            total_reward += rewards[0]  # Shared reward

            # Track coordination
            if info['n_trying'] == 1:
                successful_coords += 1
            total_orders += 1

            observations = next_observations
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)
        coordination_success.append(successful_coords / total_orders)

        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_coord = np.mean(coordination_success[-20:])
            print(f"  Episode {episode+1}/{n_episodes} | Reward: {avg_reward:.3f} | "
                  f"Coordination: {avg_coord:.1%} | ε: {agent.epsilon:.3f}")

    print("\n✓ QMIX Training Complete")
    print(f"  Final coordination rate: {np.mean(coordination_success[-20:]):.1%}")
    return agent, episode_rewards


# =============================================================================
# SECTION 7: RUN DEMONSTRATIONS AND EXPLAIN OUTPUT
# =============================================================================

print("\n" + "="*80)
print("RUNNING TRAINING DEMONSTRATIONS")
print("="*80)

# Train all three algorithms
maddpg_agent, maddpg_rewards = train_maddpg_pricing()
mappo_agent, mappo_rewards = train_mappo_inventory()
qmix_agent, qmix_rewards = train_qmix_fulfillment()


# =============================================================================
# SECTION 8: ALGORITHM COMPARISON SUMMARY
# =============================================================================

print("\n" + "="*80)
print("SECTION 7: ALGORITHM COMPARISON SUMMARY")
print("="*80)
print("""
Document Reference: "Algorithm Comparison & Selection"

┌─────────────┬──────────────┬────────────┬─────────────────┬───────────────────┐
│ Algorithm   │ Policy Type  │ Learning   │ Best For        │ Retail Use Case   │
├─────────────┼──────────────┼────────────┼─────────────────┼───────────────────┤
│ MADDPG      │ Deterministic│ Off-policy │ Continuous      │ Dynamic Pricing   │
│             │ μᵢ(oᵢ)       │ (replay)   │ + Historical    │ (3 competing      │
│             │              │            │ data available  │ stores)           │
├─────────────┼──────────────┼────────────┼─────────────────┼───────────────────┤
│ MAPPO       │ Stochastic   │ On-policy  │ Cooperative +   │ Inventory Mgmt    │
│             │ πᵢ(aᵢ|oᵢ)    │ (fresh)    │ High stochastic │ (3 warehouses)    │
│             │              │            │ + Stability     │                   │
├─────────────┼──────────────┼────────────┼─────────────────┼───────────────────┤
│ QMIX        │ Discrete     │ Off-policy │ Fully coop +    │ Order Fulfillment │
│             │ Q(oᵢ,aᵢ)     │ (replay)   │ Discrete +      │ (5-store network) │
│             │              │            │ Scalable        │                   │
└─────────────┴──────────────┴────────────┴─────────────────┴───────────────────┘

KEY TAKEAWAYS (from Document):

1️⃣  CTDE PARADIGM:
    - Train with FULL information (centralized critic sees everything)
    - Execute with LOCAL information only (actors work independently)
    - "Critic sees everything during training... actors work blind but learned"

2️⃣  MADDPG STRENGTHS:
    - Sample efficient (reuses data from replay buffer)
    - Good for continuous actions
    - Handles competitive settings
    ⚠️  Risk: Can converge to bad equilibria (price wars)

3️⃣  MAPPO STRENGTHS:
    - Stable (PPO clipping prevents wild updates)
    - Natural exploration (stochastic policy)
    - Good for cooperative, stochastic environments
    ⚠️  Limitation: Less sample efficient (discards data)

4️⃣  QMIX STRENGTHS:
    - Factorization enables truly decentralized execution
    - Scales to many agents (10-100s)
    - Monotonicity guarantees greedy = optimal
    ⚠️  Limitation: Only for fully cooperative settings

5️⃣  NON-STATIONARITY:
    "Core Challenge: Each agent learns in a non-stationary environment
    because others are also learning."

    All three algorithms address this through:
    - MADDPG: Large replay buffer, soft target updates
    - MAPPO: PPO clipping, on-policy freshness
    - QMIX: Value factorization, shared reward

6️⃣  SELECTION FRAMEWORK:
    Fully Cooperative? ──YES──> QMIX ✓
           │NO
           ▼
    Continuous Actions? ──YES──> Have Offline Data? ──YES──> MADDPG
           │                              │NO
           │                              ▼
           │                            MAPPO
           │NO
           ▼
    Need Stability? ──YES──> MAPPO
           │NO
           ▼
         QMIX (if cooperative) or MAPPO
""")


# =============================================================================
# SECTION 9: KEY CONCEPTS EXPLAINED
# =============================================================================

print("\n" + "="*80)
print("SECTION 8: KEY CONCEPTS FROM THE DOCUMENT")
print("="*80)
print("""
═══════════════════════════════════════════════════════════════════════════════
CONCEPT 1: STOCHASTIC GAMES (Multi-Agent Extension of MDP)
═══════════════════════════════════════════════════════════════════════════════

Single-Agent MDP:
  Q*(s, a) = E[r + γ max_a' Q*(s', a') | s, a]

Multi-Agent Stochastic Game:
  • N agents, state space S
  • Joint action space A = A₁ × A₂ × ... × Aₙ
  • Transition: P(s'|s, a₁, ..., aₙ) - depends on ALL
  • Reward: Rᵢ(s, a₁, ..., aₙ) - also depends on ALL

The key difference: Each agent's optimal policy depends on what OTHERS do.

═══════════════════════════════════════════════════════════════════════════════
CONCEPT 2: NASH EQUILIBRIUM
═══════════════════════════════════════════════════════════════════════════════

Definition: Jᵢ(πᵢ*, π₋ᵢ*) ≥ Jᵢ(πᵢ, π₋ᵢ*) for all agents i

In plain English: No agent can improve by unilaterally changing strategy.

Retail Example - 3-Store Pricing:
  - Cooperative Nash: All stores price high → High margins for everyone
  - Competitive Nash: All stores price low → Low margins (price war)
  - Which emerges depends on learning dynamics!

═══════════════════════════════════════════════════════════════════════════════
CONCEPT 3: CTDE - CENTRALIZED TRAINING, DECENTRALIZED EXECUTION
═══════════════════════════════════════════════════════════════════════════════

The Core Insight:
  TRAINING: Use ALL information → Better learning, credit assignment
  EXECUTION: Use ONLY local info → Practical deployment, independence

Why it works:
  - Critic learns from joint (o₁, o₂, ..., oₙ, a₁, a₂, ..., aₙ)
  - Critic identifies which agent helped/hurt
  - Gradients flow to actors
  - Actors learn to approximate good behavior from local obs only

Information Flow:
  TRAIN:  o₁,o₂,o₃ ──────────> Centralized Critic ──> Gradients ──> Actors
          a₁,a₂,a₃ ─────────┘                                        │
          rewards ──────────┘                                        │
                                                                     ▼
  EXEC:   o₁ ──> Actor₁ ──> a₁  (NO access to o₂, o₃)
          o₂ ──> Actor₂ ──> a₂  (independent)
          o₃ ──> Actor₃ ──> a₃  (parallel)

═══════════════════════════════════════════════════════════════════════════════
CONCEPT 4: MONOTONICITY IN QMIX
═══════════════════════════════════════════════════════════════════════════════

Constraint: ∂Q_total / ∂Qᵢ ≥ 0

Why it matters:
  - If increasing agent i's local Q increases total Q...
  - Then agent i can greedily maximize its own Q...
  - And this automatically maximizes the global Q!

Implementation: Use abs() on mixing network weights
  w1 = torch.abs(self.w1.weight)  # Ensures non-negative

Result: TRUE decentralized execution with global optimality guarantee

═══════════════════════════════════════════════════════════════════════════════
CONCEPT 5: NON-STATIONARITY CHALLENGE
═══════════════════════════════════════════════════════════════════════════════

The Problem:
  At time t:   Agent i learns against π₋ᵢ(t)
  At time t+1: Other agents have changed to π₋ᵢ(t+1)
  → Critic trained for t is OUTDATED at t+1!

Solutions:
  - MADDPG: Large replay buffer (mix old and new), soft target updates
  - MAPPO: Fresh data only, PPO clipping limits change magnitude
  - QMIX: Shared reward makes agents less adversarial

═══════════════════════════════════════════════════════════════════════════════
CONCEPT 6: CREDIT ASSIGNMENT
═══════════════════════════════════════════════════════════════════════════════

The Problem:
  5 stores act simultaneously. Profit drops 20%. WHO caused it?

CTDE Solution:
  Centralized critic sees ALL actions → Can identify contribution of each

Techniques:
  - Advantage functions: Aᵢ(s, aᵢ) measures how much better aᵢ is than average
  - Value decomposition: Q_total = V(s) + Σᵢ Aᵢ(s, aᵢ)
  - Counterfactual reasoning: "What if agent i did something else?"
""")


# =============================================================================
# FINAL TRAINING RESULTS
# =============================================================================

print("\n" + "="*80)
print("FINAL TRAINING RESULTS")
print("="*80)

print(f"""
MADDPG on Dynamic Pricing:
  - Final Avg Reward: {np.mean(maddpg_rewards[-10:]):.4f}
  - Environment: 3 competing stores setting prices
  - Observation: Local demand history, inventory, competitor avg price
  - Action: Continuous price in [1, 10]

MAPPO on Inventory Management:
  - Final Avg Reward: {np.mean(mappo_rewards[-10:]):.4f}
  - Environment: 3 cooperating warehouses
  - Observation: Inventory, incoming shipments, predicted demand
  - Action: Continuous restock quantity

QMIX on Order Fulfillment:
  - Final Avg Reward: {np.mean(qmix_rewards[-10:]):.4f}
  - Environment: 5 stores, 1 must fulfill each order
  - Observation: Local inventory, distance to customer
  - Action: Discrete (fulfill=1 / don't=0)
""")

print("="*80)
print("IMPLEMENTATION COMPLETE")
print("="*80)
print("""
This implementation demonstrates:
✓ Three core MARL algorithms (MADDPG, MAPPO, QMIX)
✓ CTDE paradigm (centralized training, decentralized execution)
✓ Retail domain applications (pricing, inventory, fulfillment)
✓ Key concepts: Nash equilibrium, non-stationarity, credit assignment
✓ Algorithm selection based on environment characteristics

For classroom use:
1. Run this script to see training progress
2. Modify environment parameters to explore different scenarios
3. Compare algorithm behaviors on same environment
4. Discuss trade-offs (efficiency vs stability, cooperative vs competitive)
""")
