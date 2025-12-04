# AI-Driven Travel Recommendation System Under Real-Time Disruptions

**Group 6** — *AI-Driven Travel Recommendation System Under Real-Time Disruptions*

**Python version:** 3.12.5


## Short description

The codebase is split into five modules (each with a runnable app) demonstrating each major component of the pipeline.


## Table of contents

* [Quick start](#quick-start)
* [Run the modules](#run-the-modules)
* [Repository structure](#repository-structure)


## Quick start

1. Make sure you have **Python 3.12.5** installed.

2. Create a virtual environment and activate it (Windows example shown):

```bash
py -3.12 -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

> If you use macOS / Linux, activate the venv with `source venv/bin/activate`.


## Run the modules

Each module is relatively self-contained. Run them from the repository root as follows.

**Module 1 — Bayesian Risk Estimation (CLI / Flask / Streamlit depending on implementation)**

```bash
cd module-1
python app.py
```

This runs the Bayesian Network demonstration which includes the BN structure (`bn_structure.png`), CPTs (`bn_cpts.json`) and inference code.

**Module 2 — Search-based Multi-modal Route Exploration**

```bash
cd module-2
python app.py
```

This module models the travel network as a weighted graph and implements at least one uninformed search (e.g., Uniform Cost Search) and one informed search (A* with a disruption-aware heuristic). 

**Module 3 — Adaptive Travel Plan Generation (GraphPlan & POP)**

```bash
cd module-3
python -m streamlit run app.py
```

A Streamlit app that demonstrates plan generation using GraphPlan and flexible reordering using a Partial Order Planner (POP). There is also a `test_graphplan.py` test script that exercises the planner logic.

**Module 4 — Reinforcement Learning for Robust Travel Adaptation**

```bash
cd module-4

python MDP.py

or 

python qlearning.py
```

This module contains an MDP formulation (`MDP.py`) and a Q-Learning implementation (`qlearning.py`). The app runs episodes that demonstrate learned policies vs. greedy/static baselines.

**Module 5 — LLM-based Traveler Advisory & Explanation Generation**

```bash
cd module-5
streamlit run app.py
```

A Streamlit UI that demonstrates how prompt engineering (no fine-tuning) is used to generate human-trustable explanations for recommended itineraries, including risk descriptions and fallback options.


## Repository structure

```
module-1/
  ├─ app.py
  ├─ bn_cpts.json
  └─ bn_structure.png

module-2/
  └─ app.py

module-3/
  ├─ app.py
  └─ test_graphplan.py

module-4/
  ├─ MDP.py
  └─ qlearning.py

module-5/
  └─ app.py

.gitignore
README.md             
requirements.txt
```

## Contributors

- Module 1 : Suprit Naik (22CS01018), Adarsh Dhakar (22CS01040)
- Module 2 : Sayali Khamitkar (22CS01052), Sai Nikita Palisetty (22CS01050)
- Module 3 : Suprit Naik (22CS01018), Om Prakash Behera (22CS01040)
- Module 4 : Soham Jain (22CS01007), Anshul (22CS01009), Harshil Singh (22CS01015)
- Module 5 : Adarsh Dhakar (22CS01040), Kumar Utkarsh (22CS01032)

## Team Members

Soham Jain (22CS01007)
Anshul (22CS01009)
Harshil Singh (22CS01015)
Suprit Naik (22CS01018)
Adarsh Dhakar (22CS01040)
Om Prakash Behera (22CS01040)
