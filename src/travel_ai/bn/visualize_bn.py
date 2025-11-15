# src/travel_ai/bn/visualize_bn.py
import matplotlib.pyplot as plt
import networkx as nx
import os

def draw_bn(path='outputs/bn_diagram.png'):
    print("Drawing Bayesian Network diagram...")
    G = nx.DiGraph()
    parents = ['Weather','HistoryDelay','Congestion','Festival']
    for p in parents:
        G.add_edge(p, 'DisruptionRisk')
    pos = {
        'Weather': (-2,1), 'HistoryDelay': (-1,1), 'Congestion': (0,1), 'Festival': (1,1), 'DisruptionRisk': (0,-1)
    }
    plt.figure(figsize=(6,4))
    nx.draw(G, pos=pos, with_labels=True, node_size=2500, font_size=10, arrowsize=20)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    print(f"BN diagram saved to {path}")
