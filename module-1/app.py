import itertools
import json
import math
from pprint import pprint

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from pgmpy.models import BayesianModel  # NOTE: pgmpy warns that BayesianModel -> BayesianNetwork
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import VariableElimination

# ----------------------------
# Variables and state spaces
# ----------------------------
WEATHER_STATES = ["NONE", "MODERATE", "SEVERE"]
HIST_STATES = ["LOW", "MEDIUM", "HIGH"]
CONG_STATES = ["LOW", "MEDIUM", "HIGH"]
FEST_STATES = ["NO", "YES"]
DISRUPT_STATES = ["LOW", "MEDIUM", "HIGH"]

# ----------------------------
# 1) Define network structure
# ----------------------------
model = BayesianModel(
    [
        ("Weather", "Congestion"),
        ("Festival", "Congestion"),
        ("Weather", "DisruptionRisk"),
        ("HistoricalDelay", "DisruptionRisk"),
        ("Congestion", "DisruptionRisk"),
        ("Festival", "DisruptionRisk"),
    ]
)

# ----------------------------
# 2) Priors
# ----------------------------
cpd_weather = TabularCPD(
    variable="Weather",
    variable_card=len(WEATHER_STATES),
    values=[[0.6], [0.3], [0.1]],
    state_names={"Weather": WEATHER_STATES},
)

cpd_hist = TabularCPD(
    variable="HistoricalDelay",
    variable_card=len(HIST_STATES),
    values=[[0.5], [0.3], [0.2]],
    state_names={"HistoricalDelay": HIST_STATES},
)

cpd_fest = TabularCPD(
    variable="Festival",
    variable_card=len(FEST_STATES),
    values=[[0.85], [0.15]],
    state_names={"Festival": FEST_STATES},
)

# ----------------------------
# 3) Congestion CPT
# ----------------------------
def make_congestion_cpd():
    values = []
    for w in WEATHER_STATES:
        for f in FEST_STATES:
            if w == "NONE":
                base = 0.2
            elif w == "MODERATE":
                base = 0.5
            else:
                base = 0.8
            fest_bump = 0.15 if f == "YES" else 0.0
            score = min(1.0, base + fest_bump)
            p_high = 0.6 * score + 0.05
            p_low = 0.6 * (1 - score) + 0.05
            p_med = max(0.0, 1.0 - p_high - p_low)
            norm = p_low + p_med + p_high
            p_low /= norm; p_med /= norm; p_high /= norm
            values.append((p_low, p_med, p_high))
    cols = list(zip(*values))
    row_low = list(cols[0])
    row_med = list(cols[1])
    row_high = list(cols[2])
    cpd = TabularCPD(
        variable="Congestion",
        variable_card=len(CONG_STATES),
        values=[row_low, row_med, row_high],
        evidence=["Weather", "Festival"],
        evidence_card=[len(WEATHER_STATES), len(FEST_STATES)],
        state_names={"Congestion": CONG_STATES, "Weather": WEATHER_STATES, "Festival": FEST_STATES},
    )
    return cpd

cpd_congestion = make_congestion_cpd()

# ----------------------------
# 4) Disruption CPT
# ----------------------------
WEATHER_VAL = {"NONE": 0.0, "MODERATE": 0.5, "SEVERE": 1.0}
HIST_VAL = {"LOW": 0.0, "MEDIUM": 0.5, "HIGH": 1.0}
CONG_VAL = {"LOW": 0.0, "MEDIUM": 0.5, "HIGH": 1.0}
FEST_VAL = {"NO": 0.0, "YES": 1.0}

def disruption_probs_from_parents(w, h, c, f):
    w_w = 0.40; w_h = 0.25; w_c = 0.25; w_f = 0.10
    score = w_w * WEATHER_VAL[w] + w_h * HIST_VAL[h] + w_c * CONG_VAL[c] + w_f * FEST_VAL[f]
    a = 6.0; t_high = 0.6; t_low = 0.35
    def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))
    p_high = sigmoid(a * (score - t_high))
    p_low = sigmoid(a * (t_low - score))
    p_med = max(0.0, 1.0 - p_high - p_low)
    s = p_low + p_med + p_high
    p_low /= s; p_med /= s; p_high /= s
    return (p_low, p_med, p_high)

def make_disruption_cpd():
    combos = []
    for w in WEATHER_STATES:
        for h in HIST_STATES:
            for c in CONG_STATES:
                for f in FEST_STATES:
                    combos.append((w,h,c,f))
    triples = [disruption_probs_from_parents(w,h,c,f) for (w,h,c,f) in combos]
    row_low = [t[0] for t in triples]
    row_med = [t[1] for t in triples]
    row_high = [t[2] for t in triples]
    cpd = TabularCPD(
        variable="DisruptionRisk",
        variable_card=len(DISRUPT_STATES),
        values=[row_low, row_med, row_high],
        evidence=["Weather", "HistoricalDelay", "Congestion", "Festival"],
        evidence_card=[len(WEATHER_STATES), len(HIST_STATES), len(CONG_STATES), len(FEST_STATES)],
        state_names={
            "DisruptionRisk": DISRUPT_STATES,
            "Weather": WEATHER_STATES,
            "HistoricalDelay": HIST_STATES,
            "Congestion": CONG_STATES,
            "Festival": FEST_STATES,
        },
    )
    return cpd

cpd_disruption = make_disruption_cpd()

model.add_cpds(cpd_weather, cpd_hist, cpd_fest, cpd_congestion, cpd_disruption)
if not model.check_model():
    raise RuntimeError("Model validation failed (CPD shapes/evidence mismatch).")

# ----------------------------
# Draw BN
# ----------------------------
def draw_bn_graph(model, filename="bn_structure.png"):
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 5))
    nx.draw(G, pos, with_labels=True, node_size=2500, node_color="#ffdead", font_size=10, arrowsize=20)
    plt.title("Bayesian Network - Disruption Risk")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved BN structure diagram to {filename}")

draw_bn_graph(model, filename="bn_structure.png")

def print_some_cpts():
    print("\n=== Prior: Weather CPT ===")
    pprint(cpd_weather.get_values().tolist())
    print("States (Weather):", WEATHER_STATES)

    print("\n=== Prior: HistoricalDelay CPT ===")
    pprint(cpd_hist.get_values().tolist())
    print("States (HistoricalDelay):", HIST_STATES)

    print("\n=== Prior: Festival CPT ===")
    pprint(cpd_fest.get_values().tolist())
    print("States (Festival):", FEST_STATES)

    print("\n=== Example rows of Congestion CPT (columns -> Weather x Festival combos) ===")
    cols = []
    for w in WEATHER_STATES:
        for f in FEST_STATES:
            cols.append(f"W={w},F={f}")
    rows = cpd_congestion.get_values()
    print("Columns:", cols)
    print("Rows (Congestion=LOW,MED,HIGH):")
    pprint(rows.tolist()[:])

    print("\n=== Sample Disruption CPT entries (few combos) ===")
    combos = []
    for w in WEATHER_STATES:
        for h in HIST_STATES:
            for c in CONG_STATES:
                for f in FEST_STATES:
                    combos.append((w,h,c,f))
    rows = cpd_disruption.get_values()
    for idx, parent_combo in enumerate(combos[:8]):
        p_low = rows[0, idx]; p_med = rows[1, idx]; p_high = rows[2, idx]
        print(f"W={parent_combo[0]},H={parent_combo[1]},C={parent_combo[2]},F={parent_combo[3]} -> Low={p_low:.3f}, Med={p_med:.3f}, High={p_high:.3f}")

print_some_cpts()

# ----------------------------
# Inference 
# ----------------------------
infer = VariableElimination(model)

def compute_posterior(evidence):
    """
    evidence: dict e.g. {'Weather':'SEVERE', 'HistoricalDelay':'HIGH', 'Festival':'YES'}
    Returns:
      posterior: dict mapping DisruptionRisk states -> probability
      congestion_post: dict mapping Congestion states -> probability (or None if Congestion observed)
    """
    print("\n--- Running Inference ---")
    print("Evidence:", evidence)

    # Query DisruptionRisk: VariableElimination.query returns a DiscreteFactor
    factor = infer.query(variables=["DisruptionRisk"], evidence=evidence, show_progress=False)
    # factor is DiscreteFactor. Its ordering matches the DISRUPT_STATES declared above.
    if isinstance(factor, DiscreteFactor):
        vals = factor.values
        posterior = {DISRUPT_STATES[i]: float(vals[i]) for i in range(len(DISRUPT_STATES))}
    else:
        # older pgmpy versions may return dict-like; handle gracefully
        # try to coerce to mapping
        try:
            dist = factor["DisruptionRisk"]
            vals = getattr(dist, "values", None)
            if vals is None:
                vals = list(dist)
            posterior = {DISRUPT_STATES[i]: float(vals[i]) for i in range(len(DISRUPT_STATES))}
        except Exception:
            raise RuntimeError("Unexpected query return type from pgmpy: " + str(type(factor)))

    congestion_post = None
    if "Congestion" not in evidence:
        factor_c = infer.query(variables=["Congestion"], evidence=evidence, show_progress=False)
        if isinstance(factor_c, DiscreteFactor):
            vals_c = factor_c.values
            congestion_post = {CONG_STATES[i]: float(vals_c[i]) for i in range(len(CONG_STATES))}
        else:
            try:
                distc = factor_c["Congestion"]
                vals_c = getattr(distc, "values", None)
                if vals_c is None:
                    vals_c = list(distc)
                congestion_post = {CONG_STATES[i]: float(vals_c[i]) for i in range(len(CONG_STATES))}
            except Exception:
                congestion_post = None

    return posterior, congestion_post

# ----------------------------
# Example test cases
# ----------------------------
def run_examples():
    evidence_A = {"Weather": "SEVERE", "HistoricalDelay": "HIGH", "Festival": "YES"}
    post_A, cong_A = compute_posterior(evidence_A)
    print("\nPosterior DisruptionRisk (Example A):")
    pprint(post_A)
    if cong_A:
        print("Posterior Congestion (Example A):")
        pprint(cong_A)

    evidence_B = {"Weather": "SEVERE", "HistoricalDelay": "HIGH", "Festival": "YES", "Congestion": "HIGH"}
    post_B, cong_B = compute_posterior(evidence_B)
    print("\nPosterior DisruptionRisk (Example B with observed HIGH congestion):")
    pprint(post_B)

    evidence_C = {"Weather": "NONE", "HistoricalDelay": "LOW", "Festival": "NO"}
    post_C, cong_C = compute_posterior(evidence_C)
    print("\nPosterior DisruptionRisk (Example C - favorable conditions):")
    pprint(post_C)
    if cong_C:
        print("Posterior Congestion (Example C):")
        pprint(cong_C)

run_examples()

def explain_results():
    print("\n=== Explanation of results & caveats ===")
    print("See script comments for details. CPT generation was parametric; tune weights/thresholds to match data.")

def export_cpts_json(filename="bn_cpts.json"):
    """
    Export CPDs to JSON safely by converting numpy arrays and other
    non-serializable objects to JSON-friendly types.
    """
    out = {}
    for cpd in model.get_cpds():
        var = cpd.variable
        # Build a dict for this CPD
        cpddict = {
            "variable": var,
            "variable_card": int(cpd.variable_card),
            # state_names might be dict of lists, convert safely below
            "state_names": cpd.state_names.get(var, []),
            # convert the values (numpy array) to nested lists
            "values": np.array(cpd.get_values()).tolist(),
            "evidence": [str(x) for x in (cpd.variables[1:] if len(cpd.variables) > 1 else [])],
            "evidence_card": [int(x) for x in (cpd.cardinality[1:] if len(cpd.cardinality) > 1 else [])],
        }
        out[var] = cpddict

    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        # ints/floats/str/bool are OK
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        # fallback: convert to string
        return str(obj)

    safe_out = make_serializable(out)

    with open(filename, "w", encoding="utf-8") as fh:
        json.dump(safe_out, fh, indent=2, ensure_ascii=False)

    print(f"Exported CPDs to {filename}")

export_cpts_json()
explain_results()
