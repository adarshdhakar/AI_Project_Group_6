# test_graphplan.py
import app
import pytest

def reset_state():
    app.st.session_state.literal_levels = [set(app.INITIAL)]
    app.st.session_state.action_levels = []
    app.st.session_state.mutex_action_levels = []
    app.st.session_state.mutex_literal_levels = [set()]
    app.st.session_state.producers_levels = [ {lit: {"INIT"} for lit in app.st.session_state.literal_levels[0]} ]
    app.st.session_state.level = 0
    app.st.session_state.goal_reached = app.GOALS <= app.st.session_state.literal_levels[0]
    app.st.session_state.failed_actions = set()
    app.st.session_state.action_stats = {a.name: {"trials": 1, "failures": int(a.risk_estimate*1)} for a in app.ACTIONS}

def test_expand_reaches_goals():
    reset_state()
    for _ in range(4):
        app.expand_next_level()
    assert any(app.GOALS <= S for S in app.st.session_state.literal_levels)

def test_extract_plan_validity():
    reset_state()
    for _ in range(4):
        app.expand_next_level()
    assert app.st.session_state.goal_reached
    plan = app.extract_plan(app.st.session_state.level)
    assert plan is not None
    for lvl, actions in plan.items():
        for name in actions:
            action_obj = next((a for a in app.ACTIONS if a.name == name), None)
            assert action_obj is not None
            assert action_obj.pre <= app.st.session_state.literal_levels[lvl]

def test_replanning_avoids_failed_action():
    reset_state()
    for _ in range(4):
        app.expand_next_level()
    a0 = app.st.session_state.action_levels[0][0]
    app.st.session_state.failed_actions.add(a0.name)
    plan = app.extract_plan(app.st.session_state.level)
    if plan:
        for actions in plan.values():
            assert a0.name not in actions

def test_learning_updates_change_risk():
    reset_state()
    target = app.ACTIONS[0].name
    for _ in range(5):
        app.st.session_state.action_stats.setdefault(target, {"trials":0,"failures":0})
        app.st.session_state.action_stats[target]["trials"] += 1
        app.st.session_state.action_stats[target]["failures"] += 1
    old_risk = next(a for a in app.ACTIONS if a.name==target).risk_estimate
    app.apply_learning_updates()
    new_risk = next(a for a in app.ACTIONS if a.name==target).risk_estimate
    assert new_risk != old_risk
