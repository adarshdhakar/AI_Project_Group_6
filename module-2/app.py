from heapq import heappush, heappop
from collections import defaultdict
import math

# ========================================
# 1. Edge and Graph Structures
# ========================================

class Edge:
    def __init__(self, src, dst, mode, base_time, base_cost,
                 delay_p=0.0, delay_extra=0.0, comfort=1.0, safety=1.0, meta=""):

        self.src = src
        self.dst = dst
        self.mode = mode

        # base attributes
        self.base_time = float(base_time)          # nominal time
        self.base_cost = float(base_cost)

        # disruption parameters
        self.delay_prob = float(delay_p)           # probability of delay
        self.delay_extra = float(delay_extra)      # expected delay duration

        # qualitative factors
        self.comfort = float(comfort)
        self.safety = float(safety)

        self.info = meta

    def expected_time(self):
        return self.base_time + self.delay_prob * self.delay_extra

    def clone(self):
       #copy of edges so that the original graph stays intact
        return Edge(self.src, self.dst, self.mode, self.base_time,
                    self.base_cost, self.delay_prob, self.delay_extra,
                    self.comfort, self.safety, self.info)

    def __repr__(self):
        return (f"Edge({self.src}->{self.dst}, mode={self.mode}, "
                f"time={self.base_time}, cost={self.base_cost}, "
                f"p_delay={self.delay_prob}, extra={self.delay_extra})")


class MultiModalGraph:
    def __init__(self):
        self.edges = defaultdict(list)
        self.vertices = set()

    def add_edge(self, edge):
        self.edges[edge.src].append(edge)
        self.vertices.add(edge.src)
        self.vertices.add(edge.dst)

    def get_neighbors(self, node):
        return self.edges.get(node, [])

    def nodes(self):
        return list(self.vertices)

    def apply_disruption(self, src, dst, mode=None,
                          new_delay_prob=None, new_extra_delay=None, scale_time=None):
      #modifying the disrupted edges
        for e in self.edges.get(src, []):
            if e.dst == dst and (mode is None or e.mode == mode):
                if new_delay_prob is not None:
                    e.delay_prob = new_delay_prob
                if new_extra_delay is not None:
                    e.delay_extra = new_extra_delay
                if scale_time is not None:
                    e.base_time *= scale_time


# ========================================
# 2. Cost Model
# ========================================

def compute_edge_cost(edge, w):
    time_w   = w.get("time", 1.0)
    money_w  = w.get("money", 1.0)
    delay_w  = w.get("delay", 1.0)
    comfort_w = w.get("comfort", 100.0)
    safety_w  = w.get("safety", 100.0)

    t_exp = edge.expected_time()
    extra_delay = edge.delay_prob * edge.delay_extra

    discomfort = 1.0 - edge.comfort
    unsafe     = 1.0 - edge.safety

    cost = (time_w * t_exp
            + money_w * edge.base_cost
            + delay_w * extra_delay
            + comfort_w * discomfort
            + safety_w * unsafe)

    return cost


# ========================================
# 3. Uniform-Cost Search (Dijkstra)
# ========================================

def ucs(graph, start, goal, weights):
   
    pq = []
    heappush(pq, (0.0, start, None, None))  # (g, node, parent, edge_used)
    visited = {}

    while pq:
        g_cost, node, parent, edge_used = heappop(pq)

        # prune suboptimal entries
        if node in visited and g_cost > visited[node][0] + 1e-12:   # to consider the floating point precision
            continue

        visited[node] = (g_cost, parent, edge_used)

        if node == goal:
            # reconstruct optimal route
            route = []
            cur = node
            while visited[cur][2] is not None:
                route.append(visited[cur][2])
                cur = visited[cur][1]
            route.reverse()
            return route, g_cost

        for e in graph.get_neighbors(node):
            step = compute_edge_cost(e, weights)
            new_g = g_cost + step
            prev = visited.get(e.dst)
            if prev is None or new_g + 1e-12 < prev[0]:
                heappush(pq, (new_g, e.dst, node, e.clone()))

    return None, math.inf


# ========================================
# 4. Helper: Minimum Travel-Time Heuristic
# ========================================

def get_min_time_to_goal(graph, goal):

    # Reverse Dijkstra to compute the minimal base_time need to reach the goal from every node, this is used as a heuristic for A*
    times = {n: math.inf for n in graph.nodes()}
    times[goal] = 0.0

    rev = defaultdict(list)
    for s in graph.edges:
        for e in graph.edges[s]:
            rev[e.dst].append(e)

    pq = [(0.0, goal)]
    while pq:
        t, node = heappop(pq)
        if t > times[node]:
            continue
        for inc in rev[node]:
            cand = inc.base_time + times[inc.src]
            if cand < times[inc.src]:
                times[inc.src] = cand
                heappush(pq, (cand, inc.src))

    return times


# ========================================
# 5. A* Route Search
# ========================================

def a_star(graph, start, goal, weights):
    # A* search using w_time * minimal_travel_time heuristic.
    # The heuristic used here is admissible so never over-estimates the cost and hence always finds the optimal path
    min_time = get_min_time_to_goal(graph, goal)

    def h(n):
        val = min_time.get(n, math.inf)
        if val == math.inf:
            return 0.0
        return weights.get("time", 1.0) * val

    open_list = []
    heappush(open_list, (h(start), 0.0, start, None, None))  # (f, g, node, parent, edge)
    best = {}

    while open_list:
        f, g, node, parent, used_edge = heappop(open_list)

        if node in best and g > best[node][0] + 1e-12:
            continue

        best[node] = (g, parent, used_edge)

        if node == goal:
            path = []
            cur = node
            while best[cur][2] is not None:
                path.append(best[cur][2])
                cur = best[cur][1]
            path.reverse()
            return path, g

        for e in graph.get_neighbors(node):
            step = compute_edge_cost(e, weights)
            new_g = g + step
            if e.dst not in best or new_g + 1e-12 < best[e.dst][0]:
                heappush(open_list,
                         (new_g + h(e.dst), new_g, e.dst, node, e.clone()))

    return None, math.inf


# ========================================
# 6. Demo Graph + Execution
# ========================================

def build_graph_demo():
    g = MultiModalGraph()

    # Air
    g.add_edge(Edge("A","B","flight",60,120,0.10,45,0.9,0.95))
    g.add_edge(Edge("B","A","flight",60,120,0.10,45,0.9,0.95))
    g.add_edge(Edge("B","E","flight",50,110,0.08,40,0.9,0.95))

    # Rails
    g.add_edge(Edge("A","C","train",120,50,0.05,20,0.8,0.9))
    g.add_edge(Edge("C","A","train",120,50,0.05,20,0.8,0.9))
    g.add_edge(Edge("C","E","train",100,60,0.05,25,0.8,0.9))

    # Road
    g.add_edge(Edge("B","D","cab",30,25,0.02,10,0.7,0.8))
    g.add_edge(Edge("D","E","cab",40,30,0.02,10,0.7,0.8))
    g.add_edge(Edge("C","D","cab",20,12,0.02,8,0.7,0.8))

    # Metro
    g.add_edge(Edge("A","D","metro",90,15,0.03,15,0.6,0.85))
    g.add_edge(Edge("D","A","metro",90,15,0.03,15,0.6,0.85))

    # Direct long cab
    g.add_edge(Edge("A","E","cab",240,70,0.02,20,0.6,0.75))

    return g


def describe_route(path):
    if not path:
        return "No route available."

    lines = []
    tot_time = 0.0
    tot_cost = 0.0

    for e in path:
        lines.append(f"{e.src} --[{e.mode}, time={e.base_time}m, cost={e.base_cost}]--> {e.dst}")
        tot_time += e.expected_time()
        tot_cost += e.base_cost

    return "\n".join(lines) + f"\nExpected time: {tot_time:.1f} min, Total cost: {tot_cost:.1f}"


def main():
    g = build_graph_demo()
    start, goal = "A", "E"

    # multi-factor weights
    W = {
        "time": 1.0,
        "money": 0.5,
        "delay": 1.0,
        "comfort": 40.0,
        "safety": 60.0,
    }

    print("=== BEFORE DISRUPTION ===")
    p1, c1 = ucs(g, start, goal, W)
    print("UCS (cost {:.3f})".format(c1))
    print(describe_route(p1), "\n")

    p2, c2 = a_star(g, start, goal, W)
    print("A*  (cost {:.3f})".format(c2))
    print(describe_route(p2), "\n")

    # inject disruptions
    g.apply_disruption("B","E", mode="flight", new_delay_prob=0.45, new_extra_delay=120)
    g.apply_disruption("C","E", mode="train", new_delay_prob=0.30, new_extra_delay=60)
    g.apply_disruption("A","D", mode="metro", scale_time=1.7)

    print("=== AFTER DISRUPTION ===")
    p3, c3 = ucs(g, start, goal, W)
    print("UCS (cost {:.3f})".format(c3))
    print(describe_route(p3), "\n")

    p4, c4 = a_star(g, start, goal, W)
    print("A*  (cost {:.3f})".format(c4))
    print(describe_route(p4), "\n")

    print("=== SUMMARY ===")
    print(f"Before: UCS={c1:.3f}, A*={c2:.3f}")
    print(f"After : UCS={c3:.3f}, A*={c4:.3f}")
    print("\nHeuristic: optimistic base-time estimate. UCS explores all, A* guided toward goal.")

if __name__ == "__main__":
    main()
