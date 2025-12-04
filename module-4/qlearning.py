import matplotlib.pyplot as plt
import random
import numpy as np
from collections import defaultdict

actions = ["ConfirmRoute", "SwitchRoute", "Wait", "CancelAndRebook"]

def clamp_cost(cost):
    levels = [0, 20, 40, 60, 80, 100]
    return min(levels, key=lambda x: abs(x - max(0, min(100, cost))))

def get_transitions(state, action):
    p, d, u, c = state
    if p == "ARRIVED": return [(1.0, state, 0)]
    if p == "CANCELLED": return [(1.0, state, 0)]
    
    T = []
    if action == "ConfirmRoute":
        if p == "NOT_STARTED":
            T = [(0.7, ("EN_ROUTE", "NONE", u, clamp_cost(c-20)), 5),
                 (0.2, ("EN_ROUTE", "DELAY", u, clamp_cost(c-20)), -5),
                 (0.1, ("EN_ROUTE", "CANCEL", u, clamp_cost(c-20)), -20)]
        elif p == "EN_ROUTE":
            if d == "NONE": T = [(0.8, ("ARRIVED", "NONE", u, c), 30), (0.2, ("EN_ROUTE", "DELAY", u, c), -5)]
            elif d == "DELAY": T = [(0.5, ("ARRIVED", "NONE", u, c), 20), (0.5, ("EN_ROUTE", "DELAY", u, c), -10)]
            elif d == "CANCEL": T = [(1.0, ("CANCELLED", "CANCEL", u, c), -30)]
    elif action == "SwitchRoute":
        if p == "EN_ROUTE":
            nc = clamp_cost(c-15)
            T = [(0.6, ("EN_ROUTE", "NONE", u, nc), -5), (0.3, ("EN_ROUTE", "DELAY", u, nc), -10), (0.1, ("EN_ROUTE", "CANCEL", u, nc), -25)]
    elif action == "Wait":
        nu = "HIGH" if u in ["MEDIUM", "HIGH"] else "MEDIUM"
        if d == "NONE": T = [(0.8, (p, "NONE", nu, c), -1), (0.2, (p, "DELAY", nu, c), -3)]
        elif d == "DELAY": T = [(0.7, (p, "DELAY", nu, c), -4), (0.3, (p, "CANCEL", nu, c), -20)]
        elif d == "CANCEL": T = [(1.0, ("CANCELLED", "CANCEL", nu, c), -20)]
    elif action == "CancelAndRebook":
        T = [(1.0, ("NOT_STARTED", "NONE", "LOW", clamp_cost(c-40)), -30)]
    
    return T if T else [(1.0, state, -50)] 

def step_env(state, action):
    outcomes = get_transitions(state, action)
    probs = [o[0] for o in outcomes]
    candidates = [o[1:] for o in outcomes]
    return candidates[np.random.choice(len(candidates), p=probs)]

def train_and_track():
    Q = defaultdict(float)
    epsilon = 1.0
    alpha = 0.1
    gamma = 0.95
    
    # Metrics
    history_success = []
    history_rewards = []
    window = 100
    recent_outcomes = [] 
    
    print("Training Agent...", end="")
    for ep in range(3000):
        state = ("NOT_STARTED", "NONE", "LOW", 100)
        done = False
        total_r = 0
        
        while not done:
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                q_vals = [Q[(state, a)] for a in actions]
                max_q = max(q_vals)
                best = [actions[i] for i, q in enumerate(q_vals) if q == max_q]
                action = random.choice(best)
            
            next_s, r = step_env(state, action)
            
            # Update
            old = Q[(state, action)]
            next_max = max([Q[(next_s, a)] for a in actions]) if next_s[0] not in ["ARRIVED", "CANCELLED"] else 0
            Q[(state, action)] = old + alpha * (r + gamma * next_max - old)
            
            state = next_s
            total_r += r
            if state[0] in ["ARRIVED", "CANCELLED"]:
                done = True
        
        # Track stats
        is_success = 1 if state[0] == "ARRIVED" else 0
        recent_outcomes.append(is_success)
        if len(recent_outcomes) > window: recent_outcomes.pop(0)
        
        if (ep+1) % 50 == 0:
            history_success.append(sum(recent_outcomes)/len(recent_outcomes) * 100)
            history_rewards.append(total_r)
        
        epsilon = max(0.05, epsilon * 0.999) # Decay
        
    print(" Done!")
    return Q, history_success

def simulate_detailed_run(Q):
    print("\n" + "="*60)
    print("      DETAILED SIMULATION: AGENT IN ACTION")
    print("="*60)
    state = ("NOT_STARTED", "NONE", "LOW", 100)
    done = False
    step_count = 1
    
    print(f"{'STEP':<5} | {'STATE (Progress, Disruption, Urgency, Budget)':<55} | {'ACTION':<15} | {'REWARD'}")
    print("-" * 90)
    
    total_reward = 0
    while not done and step_count < 20:
        q_vals = [Q[(state, a)] for a in actions]
        best_action = actions[np.argmax(q_vals)]
        
        next_state, reward = step_env(state, best_action)
        
        print(f"{step_count:<5} | {str(state):<55} | {best_action:<15} | {reward}")
        
        state = next_state
        total_reward += reward
        step_count += 1
        
        if state[0] in ["ARRIVED", "CANCELLED"]:
            done = True
            print("-" * 90)
            status = "SUCCESS - ARRIVED SAFELY" if state[0] == "ARRIVED" else "FAILURE - CANCELLED"
            print(f"RESULT: {status}")
            print(f"FINAL STATE: {state}")
            print(f"TOTAL REWARD: {total_reward}")

if __name__ == "__main__":
    # 1. Train and get history
    Q_table, success_rate_history = train_and_track()
    
    # 2. Simulate a run in console
    simulate_detailed_run(Q_table)
    
    # 3. Plot the learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, 3000, 50), success_rate_history, label='Q-Learning Agent', color='blue')
    plt.axhline(y=100, color='g', linestyle='--', alpha=0.3, label='Perfect Success')
    plt.title('Agent Learning Curve: Success Rate over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()