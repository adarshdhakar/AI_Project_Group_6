# app.py
import streamlit as st
import graphviz
import itertools

st.set_page_config(page_title="GraphPlan (Textbook) — Travel Domain", layout="wide")

# ------------------------------
# Action model
# ------------------------------
class Action:
    def __init__(self, name, pre=None, add=None, delete=None, noop=False):
        self.name = name
        self.pre = set(pre or [])
        self.add = set(add or [])
        self.delete = set(delete or [])
        self.noop = noop  # flag for persistence actions

    def __repr__(self):
        return f"Action({self.name})"

# ------------------------------
# Travel domain 
# ------------------------------
ACTIONS = [
    Action("CheckWeather", pre=["At(Home)"], add=["WeatherChecked"]),
    Action("BookTrain", pre=["At(Home)"], add=["TrainBooked", "HasItinerary"]),
    Action("BookFlight", pre=["At(Home)", "WeatherChecked"], add=["FlightBooked", "HasItinerary"]),
    Action("BookHotel", pre=["HasItinerary"], add=["HotelBooked"]),
    Action("Fly", pre=["FlightBooked"], add=["At(Destination)"], delete=["At(Home)"]),
    Action("TakeTrain", pre=["TrainBooked"], add=["At(Destination)"], delete=["At(Home)"]),
]

INITIAL = {"At(Home)"}
GOALS = {"At(Destination)", "HotelBooked"}

# ------------------------------
# Helpers: collect domain-positive literals
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
# Mutex checks for actions
# (1) inconsistent effects
# (2) interference
# (3) competing needs (based on previous literal-level mutexes)
# ------------------------------
def inconsistent_effects(a, b):
    return bool((a.add & b.delete) or (b.add & a.delete))

def interference(a, b):
    return bool((a.delete & b.pre) or (b.delete & a.pre))

def competing_needs(a, b, prev_literal_mutex):
    # Actions a and b have competing needs if every pair (pa in a.pre, pb in b.pre)
    # is mutex in the previous literal level.
    # If an action has no preconditions, treat it as having a precondition set { } — then not competing.
    if not a.pre or not b.pre:
        return False
    for pa in a.pre:
        for pb in b.pre:
            if (pa, pb) not in prev_literal_mutex:
                # found a compatible pair of preconditions -> not competing
                return False
    # all precondition pairs were mutex -> competing needs
    return True

def action_mutex(a, b, prev_literal_mutex):
    if inconsistent_effects(a, b):
        return True
    if interference(a, b):
        return True
    if competing_needs(a, b, prev_literal_mutex):
        return True
    return False

# ------------------------------
# SESSION STATE init
# We'll keep:
# - literal_levels: list of sets (S0, S1, S2...)
# - action_levels: list of lists (A0, A1...)
# - mutex_action_levels: list of sets of (a_name, b_name) at each action level
# - mutex_literal_levels: list of sets of (p, q) at each literal level (S0, S1, ...)
# - producers_levels: list of dicts mapping literal -> set(action_names) that produce it (for that literal level)
# ------------------------------
if "literal_levels" not in st.session_state:
    # S0 = only initial positive literals
    st.session_state.literal_levels = [set(INITIAL)]
    st.session_state.action_levels = []
    st.session_state.mutex_action_levels = []
    # For S0, compute initial literal mutex as empty (no negations yet)
    st.session_state.mutex_literal_levels = [set()]
    # producers_levels[0]: who "produced" S0 literals (treat as "INIT")
    st.session_state.producers_levels = [ {lit: {"INIT"} for lit in st.session_state.literal_levels[0]} ]
    st.session_state.level = 0
    st.session_state.goal_reached = GOALS <= st.session_state.literal_levels[0]

# ------------------------------
# Expand one GraphPlan level: S_i -> A_i -> S_{i+1}
# ------------------------------
def expand_next_level():
    if st.session_state.goal_reached:
        return

    i = st.session_state.level
    curr_S = st.session_state.literal_levels[-1]
    prev_literal_mutex = st.session_state.mutex_literal_levels[-1]  # mutex set for current S_i

    # Build candidate actions: real actions + no-ops whose preconditions are satisfied in curr_S
    possible_actions = []
    for a in ACTIONS:
        if a.pre <= curr_S:
            possible_actions.append(a)

    # LEVEL-SPECIFIC NO-OPS
    noop_actions = []
    for literal in curr_S:
        if literal.startswith("¬"):
            continue

        noop = Action(
            name=f"NoOp_{i}{literal}",
            pre=[literal],
            add=[literal],
            noop=True
        )
        noop_actions.append(noop)
        possible_actions.append(noop)

    st.session_state.action_levels.append(possible_actions)

    # Action mutexes (for A_i)
    amutex = set()
    for a, b in itertools.combinations(possible_actions, 2):
        if action_mutex(a, b, prev_literal_mutex):
            amutex.add((a.name, b.name))
            amutex.add((b.name, a.name))
    st.session_state.mutex_action_levels.append(amutex)

    # Producers for S_{i+1}
    producers = {}  # literal -> set(action.name)
    next_S = set()

    # 1) Effects: for each action, its add effects produce positive literals
    for a in possible_actions:
        for eff in a.add:
            next_S.add(eff)
            producers.setdefault(eff, set()).add(a.name)
        # 2) If action deletes a positive literal p, we consider that action produces "¬p" at next level
        for d in a.delete:
            neg = f"¬{d}"
            next_S.add(neg)
            producers.setdefault(neg, set()).add(a.name)

    # Note: textbook GraphPlan does not automatically invent negations of *all* domain literals.
    # We only add negations that are actually produced by delete lists.
    # Also, we might want to keep positive literals that persist via no-op actions (they will be added by NOOPs).

    # If a positive literal is supported by a no-op, it is already present from add effects of NOOPs.
    # But to ensure consistency, also allow "Init" producers for literals that were present in S_i and persisted
    # if there's at least one no-op available (we already created NOOPs and added their effects above).
    # The producers dict now maps each literal at S_{i+1} to actions that produce it.

    # Add producers mapping for S_{i+1}
    st.session_state.producers_levels.append(producers)
    st.session_state.literal_levels.append(next_S)

    # Compute literal mutex for S_{i+1}
    lmutex = set()
    # (A) Negation pairs: if both p and ¬p present, they're mutex
    for p in list(next_S):
        if p.startswith("¬"):
            pos = p[1:]
            if pos in next_S:
                lmutex.add((p, pos))
                lmutex.add((pos, p))

    # (B) Inconsistent support: for literals p and q at level S_{i+1},
    # if every pair of producers (ap in producers[p], bq in producers[q]) are action-mutex at A_i,
    # then p and q are literal-mutex.
    for p, q in itertools.combinations(next_S, 2):
        # skip if already marked (e.g., negation)
        if (p, q) in lmutex:
            continue

        prods_p = producers.get(p, set())
        prods_q = producers.get(q, set())

        # If either has no producers (shouldn't happen), treat them as mutex
        if not prods_p or not prods_q:
            lmutex.add((p, q))
            lmutex.add((q, p))
            continue

        all_pairs_mutex = True
        for ap in prods_p:
            for bq in prods_q:
                if (ap, bq) not in st.session_state.mutex_action_levels[-1]:
                    # found at least one pair of non-mutex producers => literals not mutex
                    all_pairs_mutex = False
                    break
            if not all_pairs_mutex:
                break

        if all_pairs_mutex:
            lmutex.add((p, q))
            lmutex.add((q, p))

    st.session_state.mutex_literal_levels.append(lmutex)

    # update level counter
    st.session_state.level += 1

    # check goals: all goals must be present in next_S and not pairwise mutex among them
    if GOALS <= next_S:
        # also ensure no pair of goal literals are mutex
        goal_mutex_conflict = False
        for g1, g2 in itertools.combinations(GOALS, 2):
            if (g1, g2) in lmutex:
                goal_mutex_conflict = True
                break
        if not goal_mutex_conflict:
            st.session_state.goal_reached = True

# ------------------------------
# UI layout
# ------------------------------
st.title("GraphPlan — Textbook Semantics (Travel Domain)")
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

st.sidebar.markdown("### Info")
st.sidebar.write(f"Levels: {len(st.session_state.literal_levels)-1} expansions")
st.sidebar.write("Goal reached:" , st.session_state.goal_reached)

# ------------------------------
# Text summary of layers (left column)
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
                # mark no-ops subtly
                nm = "(noop)" if a.noop else ""
                color = "orange" if any((a.name, x.name) in st.session_state.mutex_action_levels[i] for x in A) else "black"
                st.markdown(f"<span style='color:{color}'>{a.name} {nm}</span>", unsafe_allow_html=True)

# ------------------------------
# Graphviz visualization (textbook layout L->R)
# ------------------------------
st.header("Graph Visualization ")

dot = graphviz.Digraph(engine="dot")
dot.attr(rankdir="LR", fontsize="10")

LIT_COLOR = "#C7E9FF"
ACT_COLOR = "#BFE9B6"
NOOP_COLOR = "#EFEFEF"
MUTEX_COLOR = "red"
EDGE_COLOR = "black"

# Draw literal levels
for lvl, S in enumerate(st.session_state.literal_levels):
    with dot.subgraph() as sg:
        sg.attr(rank="same")
        for l in sorted(S):
            dot.node(f"S{lvl}_{l}", label=l, shape="box", style="filled", fillcolor=LIT_COLOR, fontname="Helvetica")

# Draw action levels
for lvl, A in enumerate(st.session_state.action_levels):
    with dot.subgraph() as sg:
        sg.attr(rank="same")
        for a in A:
            color = NOOP_COLOR if a.noop else ACT_COLOR
            shape = "rectangle"
            dot.node(f"A{lvl}_{a.name}", label=a.name + (" (noop)" if a.noop else ""), shape=shape, style="filled", fillcolor=color, fontname="Helvetica")

# Connect S_i -> A_i (preconditions) and A_i -> S_{i+1} (effects)
for i, A in enumerate(st.session_state.action_levels):
    for a in A:
        for p in sorted(a.pre):
            if f"S{i}_{p}" in dot.body:  # conservative, but graphviz node existence check is not trivial
                dot.edge(f"S{i}_{p}", f"A{i}_{a.name}", color=EDGE_COLOR)
            else:
                # if pre not present as node (rare), still try to create edge to existing literal node name
                dot.edge(f"S{i}_{p}", f"A{i}_{a.name}", color=EDGE_COLOR)

        # effects: add produces positive literals; delete produces negated literals
        for eff in sorted(a.add):
            dot.edge(f"A{i}_{a.name}", f"S{i+1}_{eff}", color=EDGE_COLOR)
        for d in sorted(a.delete):
            neg = f"¬{d}"
            dot.edge(f"A{i}_{a.name}", f"S{i+1}_{neg}", color=EDGE_COLOR, style="dotted")

# Draw red curved mutex arcs for literals (within same S level)
for lvl, lm in enumerate(st.session_state.mutex_literal_levels):
    # lm corresponds to S_l (note: mutex_literal_levels[0] corresponds to S0)
    for (p, q) in lm:
        # draw only once (directed listing has both pairs) — only draw when p < q string-wise to avoid duplicate arcs
        if p < q:
            dot.edge(f"S{lvl}_{p}", f"S{lvl}_{q}", color=MUTEX_COLOR, style="bold", constraint="false")

# Draw action mutex arcs
for lvl, am in enumerate(st.session_state.mutex_action_levels):
    for (a1, a2) in am:
        if a1 < a2:
            dot.edge(f"A{lvl}_{a1}", f"A{lvl}_{a2}", color=MUTEX_COLOR, style="bold", constraint="false")

st.graphviz_chart(dot)

# ------------------------------
# Producers / debugging info (right column)
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

# ------------------------------
# Goal status
# ------------------------------
if st.session_state.goal_reached:
    st.success("Goals are present in a non-mutex way — you can start extracting a plan (POP).")
else:
    st.info("Goals not reachable without mutex conflict. Expand more levels.")
