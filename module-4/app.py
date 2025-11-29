#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Travel Planning MDP with disruptions, urgency, and remaining cost.
"""

import math
import itertools

# -----------------------------
# Travel MDP State Definition
# -----------------------------

progress_states = ["NOT_STARTED", "EN_ROUTE", "ARRIVED", "CANCELLED"]
disruption_states = ["NONE", "DELAY", "CANCEL"]
urgency_levels = ["LOW", "MEDIUM", "HIGH"]

# Build full state-space as tuples
states = []
for p in progress_states:
    for d in disruption_states:
        for u in urgency_levels:
            for cost in [0, 20, 40, 60, 80, 100]:  # discrete cost levels
                states.append((p, d, u, cost))   # (progress, disruption, urgency, cost_left)

actions = ["ConfirmRoute", "SwitchRoute", "Wait", "CancelAndRebook"]

# -------------------------------------------------------
# Transition Model  P(s' | s, a) with Reward R(s,a,s')
# -------------------------------------------------------

def clamp_cost(cost):
    # Valid discrete cost levels
    levels = [0, 20, 40, 60, 80, 100]

    # If cost is below 0 → clamp to 0
    if cost <= 0:
        return 0

    # If above max → clamp to 100
    if cost >= 100:
        return 100

    # Otherwise snap to nearest level
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


# -------------------------------------------------------
# Policy Evaluation (same as your previous code)
# -------------------------------------------------------

def policy_evaluation(policy, gamma=1.0, theta=1e-6, max_iterations=100):
    V = {s: 0.0 for s in states}
    history = []

    for it in range(max_iterations):
        V_new = V.copy()
        for s in states:

            # terminal check
            if s[0] in ["ARRIVED", "CANCELLED"]:
                V_new[s] = V[s]
                continue

            a = policy[s]
            value = 0.0
            for prob, s_next, r in transitions(s, a):
                value += prob * (r + gamma * V[s_next])
            V_new[s] = value

        delta = max(abs(V_new[s] - V[s]) for s in states)
        history.append(V_new.copy())
        V = V_new

        print(f"Iter {it+1:2d} Δ={delta:.4e}")
        if delta < theta:
            break

    return V, history


# -------------------------------------------------------
# POLICY IMPROVEMENT
# -------------------------------------------------------
def policy_improvement(V, gamma=0.95):
    policy = {}

    for s in states:

        if s[0] in ["ARRIVED", "CANCELLED"]:
            policy[s] = "Wait"   # dummy action for terminal
            continue

        best_action = None
        best_value  = -1e9

        for a in actions:
            q = 0.0
            for prob, s_next, r in transitions(s, a):
                q += prob * (r + gamma * V[s_next])
            if q > best_value:
                best_value = q
                best_action = a

        policy[s] = best_action

    return policy


# -------------------------------------------------------
# FULL POLICY ITERATION
# -------------------------------------------------------
def policy_iteration(gamma=0.95, theta=1e-6, max_iterations=30):

    policy = {s: "Wait" for s in states}

    for it in range(max_iterations):
        print(f"\n=== POLICY ITERATION {it+1} ===")
        V, _ = policy_evaluation(policy, gamma=gamma, theta=theta)
        new_policy = policy_improvement(V, gamma=gamma)
        stable = True
        for s in states:
            if new_policy[s] != policy[s]:
                stable = False
                break
        policy = new_policy
        if stable:
            print("\nPolicy stable — optimal policy found.")
            break

    return policy, V


# -----------------------------
# Example: trivial policy
# -----------------------------
if __name__ == "__main__":
    print("=== Running FULL POLICY ITERATION ===")

    optimal_policy, optimal_V = policy_iteration(gamma=0.95)

    print("\n=== Example optimal action for starting state ===")
    print("State:", ("NOT_STARTED","NONE","LOW",100))
    print("Optimal Action:", optimal_policy[("NOT_STARTED","NONE","LOW",100)])
    print("Value:", optimal_V[("NOT_STARTED","NONE","LOW",100)])
