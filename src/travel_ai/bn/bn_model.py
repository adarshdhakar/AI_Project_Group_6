# src/travel_ai/bn/bn_model.py
import csv
import os
import numpy as np

# --- Priors (names -> list of probabilities in state order) ---
PRIORS = {
    # Weather: 0=None, 1=Mild, 2=Severe
    'Weather': [0.6, 0.3, 0.1],
    # HistoryDelay: 0=Low, 1=High
    'HistoryDelay': [0.7, 0.3],
    # Congestion: 0=Low, 1=High
    'Congestion': [0.8, 0.2],
    # Festival: 0=No, 1=Yes
    'Festival': [0.9, 0.1],
}

# --- DisruptionRisk CPT (rows = DisruptionRisk states Low/Medium/High) ---
# CPT is stored as a dict where key is (weather, history, congestion, festival)
# and value is [P(Low), P(Med), P(High)] for that parent configuration.
def build_disruption_cpt():
    cpt = {}
    # iterate all parent combinations
    for w in [0,1,2]:
        for h in [0,1]:
            for c in [0,1]:
                for f in [0,1]:
                    # scoring rule (simple domain-knowledge heuristic)
                    score = 0
                    score += {0:0,1:1,2:2}[w]
                    score += 1 if h==1 else 0
                    score += 1 if c==1 else 0
                    score += 1 if f==1 else 0
                    if score <= 1:
                        probs = [0.8, 0.15, 0.05]
                    elif score == 2:
                        probs = [0.3, 0.5, 0.2]
                    elif score == 3:
                        probs = [0.1, 0.5, 0.4]
                    else:  # score >= 4
                        probs = [0.05, 0.25, 0.7]
                    cpt[(w,h,c,f)] = probs
    return cpt

# Save CPT as CSV for PDF inclusion
def save_cpt_csv(cpt, path='outputs/disruption_cpt.csv'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['Weather','HistoryDelay','Congestion','Festival','P(Low)','P(Med)','P(High)']
        w.writerow(header)
        for key, probs in sorted(cpt.items()):
            row = list(key) + probs
            w.writerow(row)
    print(f"CPT saved to {path}")

# Pure-python inference (enumeration using factorization)
def infer_pure_python(evidence, priors=PRIORS, cpt=None):
    """
    evidence: dict e.g. {'Weather':2,'HistoryDelay':1,'Congestion':1,'Festival':0}
    returns dict: {'Low':p_low, 'Medium':p_med, 'High':p_high}
    """
    if cpt is None:
        cpt = build_disruption_cpt()
    # compute P(parents)
    p_weather = priors['Weather'][evidence['Weather']]
    p_history = priors['HistoryDelay'][evidence['HistoryDelay']]
    p_congestion = priors['Congestion'][evidence['Congestion']]
    p_festival = priors['Festival'][evidence['Festival']]
    p_parents = p_weather * p_history * p_congestion * p_festival

    # For each DisruptionRisk value r (0=Low,1=Med,2=High), joint = p_parents * P(r | parents)
    probs = cpt[(evidence['Weather'], evidence['HistoryDelay'], evidence['Congestion'], evidence['Festival'])]
    joint = [p_parents * p for p in probs]
    total = sum(joint)
    if total == 0:
        raise ValueError("Total probability zero (check CPT/prior values).")
    posterior = [j / total for j in joint]
    return {'Low': posterior[0], 'Medium': posterior[1], 'High': posterior[2]}

# Optional: if pgmpy is installed, build a pgmpy model and run inference
def build_pgmpy_model(cpt):
    try:
        from pgmpy.models import BayesianModel
        from pgmpy.factors.discrete import TabularCPD
    except Exception as e:
        print("pgmpy not available:", e)
        return None

    model = BayesianModel([
        ('Weather', 'DisruptionRisk'),
        ('HistoryDelay', 'DisruptionRisk'),
        ('Congestion', 'DisruptionRisk'),
        ('Festival', 'DisruptionRisk'),
    ])
    # build CPDs
    cpd_weather = TabularCPD('Weather', 3, [[p] for p in PRIORS['Weather']])
    cpd_history = TabularCPD('HistoryDelay', 2, [[p] for p in PRIORS['HistoryDelay']])
    cpd_congestion = TabularCPD('Congestion', 2, [[p] for p in PRIORS['Congestion']])
    cpd_festival = TabularCPD('Festival', 2, [[p] for p in PRIORS['Festival']])

    # TabularCPD needs a matrix where rows correspond to DisruptionRisk states and columns correspond to parent configurations.
    # Build columns in the order of parent state iteration used earlier
    ordered_cols = []
    for w in [0,1,2]:
        for h in [0,1]:
            for c in [0,1]:
                for f in [0,1]:
                    ordered_cols.append(cpt[(w,h,c,f)])
    import numpy as np
    table = np.array(ordered_cols).T.tolist()  # shape 3 x 24
    cpd_risk = TabularCPD('DisruptionRisk', 3, table,
                          evidence=['Weather','HistoryDelay','Congestion','Festival'],
                          evidence_card=[3,2,2,2])

    model.add_cpds(cpd_weather, cpd_history, cpd_congestion, cpd_festival, cpd_risk)
    if not model.check_model():
        raise RuntimeError("pgmpy model check failed.")
    return model
