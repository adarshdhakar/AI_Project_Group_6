# app.py (enhanced)
import streamlit as st
import graphviz
import itertools
import random
from collections import defaultdict

st.set_page_config(page_title="GraphPlan (Travel) — Interactive Prototype", layout="wide")

# ------------------------------
# Action model (extended)
# ------------------------------
class Action:
    def __init__(self, name, pre=None, add=None, delete=None, noop=False, cost=0, duration=0, risk_estimate=0.0):
        self.name = name
        self.pre = set(pre or [])
        self.add = set(add or [])
        self.delete = set(delete or [])
        self.noop = noop
        self.cost = cost
        self.duration = duration
        # initial risk estimate: probability of disruption/failure (0..1)
        self.risk_estimate = risk_estimate

    def __repr__(self):
        return f"Action({self.name})"

# ------------------------------
# Travel domain (add cost/duration/risk)
# ------------------------------
ACTIONS = [
    Action("CheckWeather", pre=["At(Home)"], add=["WeatherChecked"], cost=1, duration=5, risk_estimate=0.01),
    Action("BookTrain", pre=["At(Home)"], add=["TrainBooked", "HasItinerary"], cost=10, duration=10, risk_estimate=0.02),
    Action("BookFlight", pre=["At(Home)", "WeatherChecked"], add=["FlightBooked", "HasItinerary"], cost=80, duration=15, risk_estimate=0.08),
    Action("BookHotel", pre=["HasItinerary"], add=["HotelBooked"], cost=30, duration=5, risk_estimate=0.01),
    Action("Fly", pre=["FlightBooked"], add=["At(Destination)"], delete=["At(Home)"], cost=200, duration=120, risk_estimate=0.12),
    Action("TakeTrain", pre=["TrainBooked"], add=["At(Destination)"], delete=["At(Home)"], cost=40, duration=360, risk_estimate=0.05),
]

INITIAL = {"At(Home)"}
GOALS = {"At(Destination)", "HotelBooked"}

# ------------------------------
# Domain helper
# ------------------------------
def domain_positive_literals():
    lits = set()
    lits |= set(INITIAL)
    for a in ACTIONS:
        lits |= a.pre
        lits |= a.add
        lits |= a.delete
    return lits

POS_LITERALS = domain_positive_literals()

# ------------------------------
# Mutex checks
# ------------------------------
def inconsistent_effects(a, b):
    return bool((a.add & b.delete) or (b.add & a.delete))

def interference(a, b):
    return bool((a.delete & a.pre) or (b.delete & a.pre) or (a.delete & b.pre) or (b.delete & a.pre))
    # note: conservative but ok for this prototype

def competing_needs(a, b, prev_literal_mutex):
    if not a.pre or not b.pre:
        return False
    for pa in a.pre:
        for pb in b.pre:
            if (pa, pb) not in prev_literal_mutex:
                return False
    return True

def action_mutex(a, b, prev_literal_mutex):
    if inconsistent_effects(a, b):
        return True
    # compute interference carefully: if a.delete intersects b.pre or vice-versa
    if (a.delete & b.pre) or (b.delete & a.pre):
        return True
    if competing_needs(a, b, prev_literal_mutex):
        return True
    return False

# ------------------------------
# Session state initialization
# ------------------------------
if "literal_levels" not in st.session_state:
    st.session_state.literal_levels = [set(INITIAL)]
    st.session_state.action_levels = []
    st.session_state.mutex_action_levels = []
    st.session_state.mutex_literal_levels = [set()]
    st.session_state.producers_levels = [ {lit: {"INIT"} for lit in st.session_state.literal_levels[0]} ]
    st.session_state.level = 0
    st.session_state.goal_reached = GOALS <= st.session_state.literal_levels[0]

    # Learning / disruption bookkeeping
    st.session_state.action_stats = {a.name: {"trials": 1, "failures": int(a.risk_estimate*1)} for a in ACTIONS}  # small prior
    st.session_state.failed_actions = set()  # temporarily unavailable actions (simulate disruptions)

# ------------------------------
# Expand one GraphPlan level (unchanged, but keeps risk fields separate)
# ------------------------------
def expand_next_level():
    if st.session_state.goal_reached:
        return

    i = st.session_state.level
    curr_S = st.session_state.literal_levels[-1]
    prev_literal_mutex = st.session_state.mutex_literal_levels[-1]

    # candidate actions: real + no-op whose pre are satisfied
    possible_actions = []
    for a in ACTIONS:
        if a.pre <= curr_S:
            possible_actions.append(a)

    noop_actions = []
    for literal in curr_S:
        if literal.startswith("¬"):
            continue
        noop = Action(name=f"NoOp_{i}_{literal}", pre=[literal], add=[literal], noop=True, cost=0, duration=0, risk_estimate=0.0)
        noop_actions.append(noop)
        possible_actions.append(noop)

    st.session_state.action_levels.append(possible_actions)

    # compute action mutexes
    amutex = set()
    for a, b in itertools.combinations(possible_actions, 2):
        if action_mutex(a, b, prev_literal_mutex):
            amutex.add((a.name, b.name))
            amutex.add((b.name, a.name))
    st.session_state.mutex_action_levels.append(amutex)

    # producers for next S
    producers = {}
    next_S = set()
    for a in possible_actions:
        for eff in a.add:
            next_S.add(eff)
            producers.setdefault(eff, set()).add(a.name)
        for d in a.delete:
            neg = f"¬{d}"
            next_S.add(neg)
            producers.setdefault(neg, set()).add(a.name)

    st.session_state.producers_levels.append(producers)
    st.session_state.literal_levels.append(next_S)

    # compute literal mutex
    lmutex = set()
    for p in list(next_S):
        if p.startswith("¬"):
            pos = p[1:]
            if pos in next_S:
                lmutex.add((p, pos))
                lmutex.add((pos, p))

    for p, q in itertools.combinations(next_S, 2):
        if (p, q) in lmutex:
            continue
        prods_p = producers.get(p, set())
        prods_q = producers.get(q, set())
        if not prods_p or not prods_q:
            lmutex.add((p, q)); lmutex.add((q, p)); continue
        all_pairs_mutex = True
        for ap in prods_p:
            for bq in prods_q:
                if (ap, bq) not in st.session_state.mutex_action_levels[-1]:
                    all_pairs_mutex = False
                    break
            if not all_pairs_mutex:
                break
        if all_pairs_mutex:
            lmutex.add((p, q)); lmutex.add((q, p))

    st.session_state.mutex_literal_levels.append(lmutex)
    st.session_state.level += 1

    # check goal presence
    if GOALS <= next_S:
        goal_mutex_conflict = False
        for g1, g2 in itertools.combinations(GOALS, 2):
            if (g1, g2) in lmutex:
                goal_mutex_conflict = True
                break
        if not goal_mutex_conflict:
            st.session_state.goal_reached = True

# ------------------------------
# GraphPlan extraction (POP-like): backtracking
# ------------------------------
def extract_plan(level_index):
    """
    Try to extract a plan reaching GOALS from S_level_index.
    Returns a dict: level -> set(action_names) (actions at each level chosen),
    or None if extraction failed.
    This uses producers_levels[level] mapping from literal -> producing actions at level-1.
    """
    K = level_index
    # recursion cache
    cache = {}

    def extract(goals, k):
        key = (tuple(sorted(goals)), k)
        if key in cache:
            return cache[key]
        # base
        if k == 0:
            if set(goals) <= st.session_state.literal_levels[0]:
                cache[key] = {}
                return {}
            else:
                cache[key] = None
                return None
        # find producers at level k: producers_levels[k] is mapping for S_k from actions in A_{k-1}
        prods = st.session_state.producers_levels[k]
        A_prev = st.session_state.action_levels[k-1] if k-1 < len(st.session_state.action_levels) else []
        action_names = {a.name: a for a in A_prev}
        # For each goal, gather candidate actions that produce it and are not failed
        goal_candidates = {}
        for g in goals:
            cand = set()
            for a_name in prods.get(g, set()):
                # skip INIT
                if a_name == "INIT":
                    # if literal already present in S_{k-1} we don't need an action maybe - but model treat INIT as supported
                    continue
                if a_name in st.session_state.failed_actions:
                    continue
                # confirm a_name exists in action_levels[k-1]
                if a_name in action_names:
                    cand.add(a_name)
            # if literal present in previous literal level (persisted), allow "no op via INIT" (i.e., empty requirement)
            if g in st.session_state.literal_levels[k-1]:
                cand.add("INIT")  # special marker meaning no action needed
            if not cand:
                cache[key] = None
                return None
            goal_candidates[g] = cand

        # Backtracking: pick one producer per goal producing a set of actions with no pairwise mutex
        # We'll create a list of goals and try combinations (cartesian product with pruning)
        goals_list = list(goal_candidates.keys())
        chosen = []

        def backtrack(idx, chosen_actions):
            if idx >= len(goals_list):
                # chosen_actions is a set of action names (excluding INIT)
                # verify actions are pairwise non-mutex in A_{k-1}
                chosen_list = list(chosen_actions)
                ok = True
                # build action mutex set for A_{k-1}
                amutex = st.session_state.mutex_action_levels[k-1]
                for x, y in itertools.combinations(chosen_list, 2):
                    if (x, y) in amutex or (y, x) in amutex:
                        ok = False
                        break
                if not ok:
                    return None
                # compute new goals = union of preconditions of chosen actions (INITs don't add)
                new_goals = set()
                for an in chosen_list:
                    if an == "INIT":
                        continue
                    action_obj = action_names.get(an)
                    if action_obj:
                        new_goals |= set(action_obj.pre)
                # recursion
                subplan = extract(new_goals, k-1)
                if subplan is None:
                    return None
                # append chosen actions at level k-1 into subplan
                out = dict(subplan)
                out.setdefault(k-1, set())
                out[k-1] |= set(x for x in chosen_list if x != "INIT")
                return out

            g = goals_list[idx]
            for cand in sorted(goal_candidates[g]):
                # try choose cand for goal g
                if cand == "INIT":
                    res = backtrack(idx+1, chosen_actions)
                    if res is not None:
                        return res
                else:
                    # add cand, but avoid duplicates
                    if cand in chosen_actions:
                        res = backtrack(idx+1, chosen_actions)
                        if res is not None:
                            return res
                    else:
                        chosen_actions.add(cand)
                        res = backtrack(idx+1, chosen_actions)
                        if res is not None:
                            return res
                        chosen_actions.remove(cand)
            return None

        plan = backtrack(0, set())
        cache[key] = plan
        return plan

    # start extraction from topmost level K where goals are present
    return extract(set(GOALS), K)

# ------------------------------
# Utilities: compute cost/duration and build readable plan ordering
# ------------------------------
def build_ordered_plan(plan_by_level):
    # produce a list of actions in chronological order (by level increasing)
    ordered = []
    name_to_obj = {a.name: a for a in ACTIONS}
    for lvl in sorted(plan_by_level.keys()):
        for an in sorted(plan_by_level[lvl]):
            ordered.append({
                "level": lvl,
                "name": an,
                "cost": name_to_obj.get(an).cost if name_to_obj.get(an) else 0,
                "duration": name_to_obj.get(an).duration if name_to_obj.get(an) else 0,
                "risk": name_to_obj.get(an).risk_estimate if name_to_obj.get(an) else 0.0
            })
    total_cost = sum(x["cost"] for x in ordered)
    total_duration = sum(x["duration"] for x in ordered)
    return ordered, total_cost, total_duration

# ------------------------------
# Simulate disruption and learning
# ------------------------------
def simulate_disruption(action_name=None):
    """
    Simulate running action_name (or random chosen action from planned sequence).
    Use action's current risk estimate to sample failure. Update action_stats and failed_actions if failure.
    """
    # choose an action if not provided
    if not action_name:
        # choose random non-noop action
        candidates = [a for a in ACTIONS if not a.noop]
        action_obj = random.choice(candidates)
    else:
        action_obj = next((a for a in ACTIONS if a.name == action_name), None)
        if not action_obj:
            return {"result": "unknown action"}

    name = action_obj.name
    st.session_state.action_stats.setdefault(name, {"trials": 0, "failures": 0})
    st.session_state.action_stats[name]["trials"] += 1

    # effective risk: combine static estimate with empirical (trials/failures smoothing)
    trials = st.session_state.action_stats[name]["trials"]
    failures = st.session_state.action_stats[name]["failures"]
    emp_risk = failures / trials if trials>0 else action_obj.risk_estimate
    # simple blend: 0.6 empirical, 0.4 prior
    eff_risk = 0.6 * emp_risk + 0.4 * action_obj.risk_estimate

    failed = random.random() < eff_risk
    if failed:
        st.session_state.action_stats[name]["failures"] += 1
        st.session_state.failed_actions.add(name)
        return {"result": "failed", "action": name, "eff_risk": eff_risk}
    else:
        return {"result": "ok", "action": name, "eff_risk": eff_risk}

def apply_learning_updates():
    # update each in-memory action object's risk_estimate with smoothed empirical estimate
    for a in ACTIONS:
        stats = st.session_state.action_stats.get(a.name, None)
        if not stats:
            continue
        # Laplace smoothing
        trials = stats["trials"]
        failures = stats["failures"]
        a.risk_estimate = (failures + 1) / (trials + 2)  # smoothed
    st.success("Updated action risk estimates from observed stats (Laplace-smoothed).")

# ------------------------------
# UI layout
# ------------------------------
st.title("Adaptive Travel Planning — GraphPlan + POP prototype")
st.write("Initial:", INITIAL, " Goals:", GOALS)

st.sidebar.header("Controls")
if st.sidebar.button("➕ Expand next level"):
    expand_next_level()

if st.sidebar.button("🔄 Reset"):
    st.session_state.literal_levels = [set(INITIAL)]
    st.session_state.action_levels = []
    st.session_state.mutex_action_levels = []
    st.session_state.mutex_literal_levels = [set()]
    st.session_state.producers_levels = [ {lit: {"INIT"} for lit in st.session_state.literal_levels[0]} ]
    st.session_state.level = 0
    st.session_state.goal_reached = GOALS <= st.session_state.literal_levels[0]
    st.session_state.failed_actions = set()
    st.session_state.action_stats = {a.name: {"trials": 1, "failures": int(a.risk_estimate*1)} for a in ACTIONS}

st.sidebar.markdown("### Planner actions")
if st.sidebar.button("🔎 Extract plan (POP)"):
    if not st.session_state.goal_reached:
        st.sidebar.warning("Goals not yet reachable — expand more levels.")
    else:
        plan = extract_plan(st.session_state.level)
        st.session_state.last_extracted_plan = plan

if st.sidebar.button("⚠️ Simulate disruption (random)"):
    result = simulate_disruption()
    if result["result"] == "failed":
        st.sidebar.error(f"Simulated failure of {result['action']} (eff risk {result['eff_risk']:.2f}). Action marked unavailable for replanning.")
    else:
        st.sidebar.success(f"Simulated success of {result['action']} (eff risk {result['eff_risk']:.2f}).")

if st.sidebar.button("🏫 Apply learning update to risk estimates"):
    apply_learning_updates()

st.sidebar.markdown("---")
st.sidebar.write(f"Levels: {len(st.session_state.literal_levels)-1} expansions")
st.sidebar.write("Goal reached:" , st.session_state.goal_reached)
st.sidebar.write("Failed actions (simulated):", sorted(st.session_state.failed_actions))
st.sidebar.markdown("Action stats (trials, failures):")
for name, stats in st.session_state.action_stats.items():
    st.sidebar.write(f"{name}: {stats['trials']}, {stats['failures']}")

# ------------------------------
# Left column: planning graph & text
# ------------------------------
st.header("Planning Graph Levels (text)")
col1, col2 = st.columns([1, 2])
with col1:
    for i, S in enumerate(st.session_state.literal_levels):
        st.markdown(f"#### S{i} (literals)")
        if S:
            for l in sorted(S):
                color = "red" if any((l, x) in st.session_state.mutex_literal_levels[i] for x in S) else "black"
                st.markdown(f"<span style='color:{color}'>{l}</span>", unsafe_allow_html=True)
        else:
            st.write("— none —")
        if i < len(st.session_state.action_levels):
            st.markdown(f"#### A{i} (actions)")
            A = st.session_state.action_levels[i]
            for a in A:
                nm = "(noop)" if a.noop else ""
                color = "gray" if a.name in st.session_state.failed_actions else "orange" if a.noop else "black"
                st.markdown(f"<span style='color:{color}'>{a.name} {nm}</span>", unsafe_allow_html=True)

# ------------------------------
# Graphviz visualization
# ------------------------------
st.header("Graph Visualization")
dot = graphviz.Digraph(engine="dot")
dot.attr(rankdir="LR", fontsize="10")
LIT_COLOR = "#C7E9FF"
ACT_COLOR = "#BFE9B6"
NOOP_COLOR = "#EFEFEF"
MUTEX_COLOR = "red"
EDGE_COLOR = "black"

for lvl, S in enumerate(st.session_state.literal_levels):
    with dot.subgraph() as sg:
        sg.attr(rank="same")
        for l in sorted(S):
            dot.node(f"S{lvl}_{l}", label=l, shape="box", style="filled", fillcolor=LIT_COLOR, fontname="Helvetica")

for lvl, A in enumerate(st.session_state.action_levels):
    with dot.subgraph() as sg:
        sg.attr(rank="same")
        for a in A:
            color = NOOP_COLOR if a.noop else ACT_COLOR
            if a.name in st.session_state.failed_actions:
                color = "#f8d7da"  # pale red for failed
            dot.node(f"A{lvl}_{a.name}", label=a.name + (" (noop)" if a.noop else ""), shape="rectangle", style="filled", fillcolor=color, fontname="Helvetica")

for i, A in enumerate(st.session_state.action_levels):
    for a in A:
        for p in sorted(a.pre):
            dot.edge(f"S{i}_{p}", f"A{i}_{a.name}", color=EDGE_COLOR)
        for eff in sorted(a.add):
            dot.edge(f"A{i}_{a.name}", f"S{i+1}_{eff}", color=EDGE_COLOR)
        for d in sorted(a.delete):
            neg = f"¬{d}"
            dot.edge(f"A{i}_{a.name}", f"S{i+1}_{neg}", color=EDGE_COLOR, style="dotted")

for lvl, lm in enumerate(st.session_state.mutex_literal_levels):
    for (p, q) in lm:
        if p < q:
            dot.edge(f"S{lvl}_{p}", f"S{lvl}_{q}", color=MUTEX_COLOR, style="bold", constraint="false")

for lvl, am in enumerate(st.session_state.mutex_action_levels):
    for (a1, a2) in am:
        if a1 < a2:
            dot.edge(f"A{lvl}_{a1}", f"A{lvl}_{a2}", color=MUTEX_COLOR, style="bold", constraint="false")

st.graphviz_chart(dot)

# ------------------------------
# Right column: producers, extracted plan, & replanning UI
# ------------------------------
with col2:
    st.subheader("Producers & Mutex Info")
    for lvl, producers in enumerate(st.session_state.producers_levels):
        st.markdown(f"**Producers for S{lvl}**")
        if not producers:
            st.write("— none —")
            continue
        for lit in sorted(producers.keys()):
            st.write(f"{lit}  <-  {sorted(producers[lit])}")

    st.markdown("---")
    st.write("Action-level mutexes (sample):")
    for i, am in enumerate(st.session_state.mutex_action_levels):
        st.write(f"A{i} mutex pairs: {sorted(set(tuple(sorted(x)) for x in am))}")

    st.markdown("---")
    st.write("Literal-level mutexes (sample):")
    for i, lm in enumerate(st.session_state.mutex_literal_levels):
        st.write(f"S{i} mutex pairs: {sorted(set(tuple(sorted(x)) for x in lm)) if lm else '[]'}")

    st.markdown("---")
    st.subheader("Plan extraction & replanning")
    if "last_extracted_plan" in st.session_state and st.session_state.last_extracted_plan:
        plan_map = st.session_state.last_extracted_plan
        ordered, total_cost, total_duration = build_ordered_plan(plan_map)
        st.markdown("**Extracted Plan (chronological)**")
        for step in ordered:
            st.write(f"Lvl {step['level']}: {step['name']} — cost {step['cost']}, dur {step['duration']} min, risk {step['risk']:.2f}")
        st.write(f"**Total cost:** {total_cost}  —  **Total duration:** {total_duration} min")
        st.markdown("**Explanation:**")
        st.write("This plan was extracted from the planning graph levels using a POP-style backtracking extractor. Actions marked in pale red have been simulated as failing and were avoided during extraction.")
    else:
        st.info("No plan extracted yet. Expand levels until goals are reachable, then click 'Extract plan (POP)'.")

    st.markdown("---")
    st.subheader("Simulate disruption on a specific action")
    select_action = st.selectbox("Action to simulate", options=["<random>"] + [a.name for a in ACTIONS])
    if st.button("Simulate this action"):
        if select_action == "<random>":
            res = simulate_disruption()
        else:
            res = simulate_disruption(select_action)
        if res["result"] == "failed":
            st.error(f"Action {res['action']} failed (eff risk {res['eff_risk']:.2f}). It is now unavailable for replanning.")
        else:
            st.success(f"Action {res['action']} succeeded (eff risk {res['eff_risk']:.2f}).")

    st.markdown("---")
    st.subheader("Notes / Limitations")
    st.write("""
    - This prototype demonstrates: GraphPlan expansion, a POP-style extractor, simulated disruptions and a tiny online learning update for per-action risk estimates.
    - Missing pieces for a production-ready 'AI-Driven Travel Recommendation System':
      1. Real-time feeds (weather, flights/trains/cab APIs) to update literals and actions dynamically.
      2. Temporal scheduling (time windows, precise start/finish times), capacity constraints.
      3. Multi-objective optimizer to trade safety vs cost vs duration (e.g., weighted scoring).
      4. More advanced learning (Bayesian updating, contextual risk models based on route/season).
      5. Natural-language explanations with transparency for end-users.
    - Next recommended steps: integrate real APIs, add an explicit temporal model, and replace greedy extraction with a planner optimized on cost+risk (A*/MCTS or MILP).
    """)

