#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Travel Planning MDP with disruptions, urgency, and remaining cost.
Added: Q-Learning training + evaluation and comparison to a greedy static itinerary and DP optimal policy.
"""

import math
import random
import itertools
from collections import defaultdict

# -----------------------------
# MDP definitions (from your file)
# -----------------------------

progress_states = ["NOT_STARTED", "EN_ROUTE", "ARRIVED", "CANCELLED"]
disruption_states = ["NONE", "DELAY", "CANCEL"]
urgency_levels = ["LOW", "MEDIUM", "HIGH"]

states = []
for p in progress_states:
    for d in disruption_states:
        for u in urgency_levels:
            for cost in [0, 20, 40, 60, 80, 100]:
                states.append((p, d, u, cost))

actions = ["ConfirmRoute", "SwitchRoute", "Wait", "CancelAndRebook"]

def clamp_cost(cost):
    levels = [0, 20, 40, 60, 80, 100]
    if cost <= 0:
        return 0
    if cost >= 100:
        return 100
    return min(levels, key=lambda x: abs(x - cost))

def transitions(state, action):
    progress, disruption, urgency, cost_left = state

    if progress == "ARRIVED":
        return [(1.0, state, 0.0)]

    if progress == "CANCELLED":
        return [(1.0, state, -10.0)]

    T = []

    # ConfirmRoute
    if action == "ConfirmRoute":
        if progress == "NOT_STARTED":
            c = clamp_cost(cost_left - 20)
            T.append((0.7, ("EN_ROUTE", "NONE",  urgency, c),  +5))
            T.append((0.2, ("EN_ROUTE", "DELAY", urgency, c),  -5))
            T.append((0.1, ("EN_ROUTE", "CANCEL",urgency, c), -20))

        elif progress == "EN_ROUTE":
            if disruption == "NONE":
                T.append((0.8, ("ARRIVED", "NONE", urgency, cost_left), +30))
                T.append((0.2, ("EN_ROUTE","DELAY",urgency, cost_left), -5))

            elif disruption == "DELAY":
                T.append((0.5, ("ARRIVED", "NONE", urgency, cost_left), +20))
                T.append((0.5, ("EN_ROUTE","DELAY",urgency, cost_left), -10))

            elif disruption == "CANCEL":
                T.append((1.0, ("CANCELLED","CANCEL",urgency,cost_left), -30))

        return T

    # SwitchRoute
    if action == "SwitchRoute":
        if progress == "EN_ROUTE":
            c = clamp_cost(cost_left - 15)
            T.append((0.6, ("EN_ROUTE","NONE", urgency, c), -5))
            T.append((0.3, ("EN_ROUTE","DELAY",urgency, c), -10))
            T.append((0.1, ("EN_ROUTE","CANCEL",urgency, c), -25))

        else:
            c = clamp_cost(cost_left - 10)
            T.append((1.0, ("NOT_STARTED","NONE", urgency, c), -2))

        return T

    # Wait
    if action == "Wait":
        new_urgency = {
            "LOW": "MEDIUM",
            "MEDIUM": "HIGH",
            "HIGH": "HIGH"
        }[urgency]

        if disruption == "NONE":
            T.append((0.8, (progress,"NONE", new_urgency, cost_left), -1))
            T.append((0.2, (progress,"DELAY",new_urgency, cost_left), -3))

        elif disruption == "DELAY":
            T.append((0.7, (progress,"DELAY", new_urgency, cost_left), -4))
            T.append((0.3, (progress,"CANCEL",new_urgency, cost_left), -20))

        elif disruption == "CANCEL":
            T.append((1.0, ("CANCELLED","CANCEL",new_urgency,cost_left), -20))

        return T

    # CancelAndRebook
    if action == "CancelAndRebook":
        new_cost = clamp_cost(cost_left - 40)
        T.append((1.0, ("NOT_STARTED","NONE","LOW",new_cost), -30))
        return T

    raise ValueError("Invalid state/action")


# -----------------------------
# Terminal check and sampling
# -----------------------------
def is_terminal(state):
    return state[0] in ("ARRIVED", "CANCELLED")

def sample_next_state(state, action):
    """Sample a single next state according to transition probabilities."""
    trans = transitions(state, action)
    r = random.random()
    cumulative = 0.0
    for prob, s_next, reward in trans:
        cumulative += prob
        if r <= cumulative:
            return s_next, reward
    # numerical fallback
    return trans[-1][1], trans[-1][2]


# -------------------------------------------------------
# Policy evaluation & improvement (keep your functions)
# (slightly adapted to be callable here)
# -------------------------------------------------------
def policy_evaluation(policy, gamma=0.95, theta=1e-6, max_iterations=200):
    V = {s: 0.0 for s in states}
    for _ in range(max_iterations):
        delta = 0.0
        V_new = V.copy()
        for s in states:
            if is_terminal(s):
                V_new[s] = 0.0
                continue
            a = policy.get(s, "Wait")
            val = 0.0
            for prob, s_next, r in transitions(s, a):
                val += prob * (r + gamma * V[s_next])
            delta = max(delta, abs(val - V[s]))
            V_new[s] = val
        V = V_new
        if delta < theta:
            break
    return V

def policy_improvement(V, gamma=0.95):
    policy = {}
    for s in states:
        if is_terminal(s):
            policy[s] = "Wait"
            continue
        best_a = None
        best_q = -1e12
        for a in actions:
            q = 0.0
            for prob, s_next, r in transitions(s, a):
                q += prob * (r + gamma * V[s_next])
            if q > best_q:
                best_q = q
                best_a = a
        policy[s] = best_a
    return policy

def policy_iteration(gamma=0.95):
    policy = {s: "Wait" for s in states}
    for _ in range(50):
        V = policy_evaluation(policy, gamma=gamma)
        new_policy = policy_improvement(V, gamma=gamma)
        if all(new_policy[s] == policy[s] for s in states):
            break
        policy = new_policy
    return policy, V


# -----------------------------
# Q-Learning
# -----------------------------
def q_learning(num_episodes=2000, alpha=0.1, gamma=0.95, epsilon_start=0.3, epsilon_end=0.02, seed=0, max_steps=200):
    random.seed(seed)
    Q = defaultdict(float)  # Q[(state, action)] = value
    episode_returns = []

    for ep in range(1, num_episodes + 1):
        state = ("NOT_STARTED", "NONE", "LOW", 100)
        total_return = 0.0
        discount = 1.0
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * (1 - (ep - 1) / num_episodes)  # linear decay

        for step in range(max_steps):
            # epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                # choose argmax_a Q(s,a)
                vals = [(Q[(state, a)], a) for a in actions]
                max_val = max(vals)[0]
                # tie-breaker random among best
                best_as = [a for v, a in vals if v == max_val]
                action = random.choice(best_as)

            next_state, reward = sample_next_state(state, action)
            total_return += discount * reward

            # Q update
            q_sa = Q[(state, action)]
            # estimate of next state's best value
            next_vals = [Q[(next_state, a)] for a in actions]
            q_target = reward + gamma * max(next_vals) if not is_terminal(next_state) else reward
            Q[(state, action)] = q_sa + alpha * (q_target - q_sa)

            state = next_state
            discount *= gamma

            if is_terminal(state):
                break

        episode_returns.append(total_return)
        # occasional progress printing
        if ep % max(1, num_episodes // 10) == 0:
            avg_last = sum(episode_returns[-(num_episodes//10):]) / max(1, (num_episodes//10))
            print(f"[Q-learning] Episode {ep}/{num_episodes} epsilon={epsilon:.3f} avg_recent_return={avg_last:.3f}")

    # derive greedy policy from Q
    learned_policy = {}
    for s in states:
        if is_terminal(s):
            learned_policy[s] = "Wait"
            continue
        best_a = max(actions, key=lambda a: Q[(s, a)])
        learned_policy[s] = best_a

    return Q, learned_policy, episode_returns


# -----------------------------
# Baseline greedy static itinerary policy
# -----------------------------
def greedy_static_policy(state):
    """
    Simple static itinerary rules (deterministic):
    - If NOT_STARTED: try ConfirmRoute (book and go)
    - If EN_ROUTE:
        - if disruption == CANCEL -> CancelAndRebook
        - if disruption == DELAY and urgency HIGH -> SwitchRoute else ConfirmRoute
    - If arrived/cancelled: Wait (terminal)
    """
    progress, disruption, urgency, cost_left = state
    if progress == "ARRIVED" or progress == "CANCELLED":
        return "Wait"
    if progress == "NOT_STARTED":
        return "ConfirmRoute"
    if progress == "EN_ROUTE":
        if disruption == "CANCEL":
            return "CancelAndRebook"
        if disruption == "DELAY":
            if urgency == "HIGH":
                return "SwitchRoute"
            else:
                return "Wait"  # be conservative and wait if delay and not urgent
        return "ConfirmRoute"
    return "Wait"


# -----------------------------
# Evaluation utilities
# -----------------------------
def run_episode_with_policy(policy_fn, max_steps=200, gamma=0.95, seed=None):
    if seed is not None:
        random.seed(seed)
    state = ("NOT_STARTED", "NONE", "LOW", 100)
    total_return = 0.0
    discount = 1.0
    steps = 0
    final_state = state
    for _ in range(max_steps):
        steps += 1
        action = policy_fn(state) if callable(policy_fn) else policy_fn.get(state, "Wait")
        next_state, reward = sample_next_state(state, action)
        total_return += discount * reward
        discount *= gamma
        state = next_state
        final_state = state
        if is_terminal(state):
            break
    success = 1 if final_state[0] == "ARRIVED" else 0
    return {"return": total_return, "steps": steps, "final_state": final_state, "success": success}

def evaluate_policy_fn(policy_fn, n_episodes=200, gamma=0.95, seed=123):
    returns = []
    successes = 0
    steps_acc = 0
    for i in range(n_episodes):
        res = run_episode_with_policy(policy_fn, gamma=gamma, seed=seed + i)
        returns.append(res["return"])
        successes += res["success"]
        steps_acc += res["steps"]
    return {
        "avg_return": sum(returns) / len(returns),
        "success_rate": successes / n_episodes,
        "avg_steps": steps_acc / n_episodes
    }


# -----------------------------
# Main demonstration
# -----------------------------
def main():
    random.seed(0)
    print("=== Running policy iteration (DP) to get reference optimal policy (may take a moment) ===")
    dp_policy, dp_V = policy_iteration(gamma=0.95)
    print("DP policy computed.")

    print("\n=== Training Q-Learning agent ===")
    Q, q_policy, returns = q_learning(num_episodes=2000, alpha=0.1, gamma=0.95, epsilon_start=0.3, epsilon_end=0.02, seed=1)
    print("Q-Learning training finished.")

    print("\n=== Evaluating policies (200 episodes each) ===")
    n_eval = 200
    # Greedy static baseline (callable)
    static_metrics = evaluate_policy_fn(greedy_static_policy, n_episodes=n_eval, gamma=0.95, seed=10)
    print("Static greedy policy:", static_metrics)

    # Learned Q policy (mapping dict)
    q_metrics = evaluate_policy_fn(lambda s: q_policy[s], n_episodes=n_eval, gamma=0.95, seed=200)
    print("Q-Learned greedy policy:", q_metrics)

    # DP policy (mapping dict)
    dp_metrics = evaluate_policy_fn(lambda s: dp_policy[s], n_episodes=n_eval, gamma=0.95, seed=400)
    print("DP (value-iteration/policy-iteration) derived policy:", dp_metrics)

    print("\n=== Summary (higher avg_return and success_rate better) ===")
    print(f"Static greedy: avg_return={static_metrics['avg_return']:.2f}, success_rate={static_metrics['success_rate']:.2f}, avg_steps={static_metrics['avg_steps']:.1f}")
    print(f"Q-learned:     avg_return={q_metrics['avg_return']:.2f}, success_rate={q_metrics['success_rate']:.2f}, avg_steps={q_metrics['avg_steps']:.1f}")
    print(f"DP-optimal:    avg_return={dp_metrics['avg_return']:.2f}, success_rate={dp_metrics['success_rate']:.2f}, avg_steps={dp_metrics['avg_steps']:.1f}")

    # show a few sample runs to inspect behavior
    print("\nSample runs (policy -> final state):")
    for label, policy in [("Static", greedy_static_policy), ("Q", lambda s: q_policy[s]), ("DP", lambda s: dp_policy[s])]:
        res = run_episode_with_policy(policy, seed=999)
        print(f"{label}: final_state={res['final_state']}, return={res['return']:.2f}, steps={res['steps']}")

if __name__ == "__main__":
    main()
