"""
Ant Colony Optimization (ACO) for Network Routing
Assignment 1 - PS13 - Communication Networks
BITS Pilani - MTech AI/ML - S2 2025-2026
Group: G093

This program implements:
1. ACO algorithm for finding efficient routing paths in a network
2. Dijkstra's algorithm for deterministic shortest path comparison
3. Analysis of two hyperparameter configurations and their effects

Usage:
    python aco_routing_PS13.py

Input:  inputPS13.txt  (network topology with source/destination)
Output: outputPS13.txt (routing results and analysis)
"""

import random
import heapq
import sys


# ============================================================
# Graph Data Structure
# ============================================================
class Graph:
    """
    Weighted undirected graph representing a network of routers.
    Uses adjacency list for efficient neighbor lookups.
    """

    def __init__(self):
        """Initialize an empty graph."""
        self.adj = {}        # {node: {neighbor: latency}}
        self.nodes = set()

    def add_node(self, node):
        """Add a router node to the network."""
        if node in self.nodes:
            return
        self.adj[node] = {}
        self.nodes.add(node)

    def add_edge(self, u, v, weight):
        """
        Add a bidirectional communication link between two routers.
        Validates that edge weight (latency) is positive.
        """
        if weight <= 0:
            raise ValueError(f"Edge weight must be positive, got {weight}")
        self.add_node(u)
        self.add_node(v)
        self.adj[u][v] = weight
        self.adj[v][u] = weight

    def get_neighbors(self, node):
        """
        Return dict of {neighbor: weight} for a given node.
        Raises KeyError if node not in graph.
        """
        if node not in self.adj:
            raise KeyError(f"Node {node} not found in the network")
        return self.adj[node]

    def get_weight(self, u, v):
        """Get latency between two routers. Returns inf if no direct link."""
        if u in self.adj and v in self.adj[u]:
            return self.adj[u][v]
        return float('inf')


# ============================================================
# Ant Colony Optimization Algorithm
# ============================================================
class AntColonyOptimization:
    """
    ACO algorithm for finding minimum-latency routing paths.

    Ants traverse the network probabilistically, choosing next nodes
    based on pheromone levels (tau) and heuristic desirability (eta = 1/latency).
    Over iterations, pheromone trails reinforce efficient routes.

    Probability rule:
        P(i,j) = [tau(i,j)^alpha * eta(i,j)^beta] / sum_k[tau(i,k)^alpha * eta(i,k)^beta]

    Pheromone update:
        tau(i,j) = (1-rho) * tau(i,j) + sum_k(delta_tau_k)
        delta_tau_k = 1/L_k  if ant k used edge (i,j), else 0
    """

    def __init__(self, graph, num_ants=10, alpha=1.0, beta=2.0,
                 rho=0.5, num_iterations=100, initial_pheromone=1.0):
        """
        Initialize ACO with given configuration.

        Args:
            graph:             Network Graph object
            num_ants:          Number of ants per iteration
            alpha:             Pheromone importance exponent
            beta:              Heuristic importance exponent
            rho:               Pheromone evaporation rate (0 < rho < 1)
            num_iterations:    Number of iterations
            initial_pheromone: Starting pheromone level on all edges
        """
        self.graph = graph
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.num_iterations = num_iterations

        # Initialize pheromone matrix on all edges
        self.pheromone = {}
        for node in graph.nodes:
            self.pheromone[node] = {}
            for neighbor in graph.get_neighbors(node):
                self.pheromone[node][neighbor] = initial_pheromone

    def _heuristic(self, u, v):
        """Heuristic desirability: inverse of edge latency. eta(u,v) = 1/latency(u,v)"""
        w = self.graph.get_weight(u, v)
        if w <= 0 or w == float('inf'):
            return 0.0
        return 1.0 / w

    def _select_next_node(self, current, visited):
        """
        Select the next node using ACO probability rule (roulette wheel selection).
        Returns None if all neighbors have been visited (dead end).
        """
        neighbors = self.graph.get_neighbors(current)
        unvisited = {n: w for n, w in neighbors.items() if n not in visited}

        if not unvisited:
            return None  # Dead end - no unvisited neighbors

        # Calculate unnormalized probabilities for each unvisited neighbor
        probabilities = {}
        total = 0.0
        for neighbor in unvisited:
            tau = self.pheromone[current][neighbor] ** self.alpha
            eta = self._heuristic(current, neighbor) ** self.beta
            prob = tau * eta
            probabilities[neighbor] = prob
            total += prob

        if total == 0:
            # Fallback: uniform random selection among unvisited
            return random.choice(list(unvisited.keys()))

        # Roulette wheel selection
        r = random.random() * total
        cumulative = 0.0
        for neighbor, prob in probabilities.items():
            cumulative += prob
            if cumulative >= r:
                return neighbor
        return list(probabilities.keys())[-1]

    def _construct_path(self, source, destination):
        """
        A single ant constructs a path from source to destination.

        Returns:
            (path, cost): path is list of nodes, cost is total latency.
                          Returns (None, inf) if ant reaches a dead end.
        """
        path = [source]
        visited = {source}
        current = source

        while current != destination:
            next_node = self._select_next_node(current, visited)
            if next_node is None:
                return None, float('inf')
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        # Sum edge weights along the path
        total_cost = sum(
            self.graph.get_weight(path[i], path[i + 1])
            for i in range(len(path) - 1)
        )
        return path, total_cost

    def _update_pheromones(self, ant_paths):
        """
        Update pheromone levels: evaporation followed by deposit.
        Evaporation: tau(i,j) *= (1 - rho)
        Deposit:     tau(i,j) += 1/L_k for each ant k that used edge (i,j)
        """
        # Evaporation on all edges
        for u in self.pheromone:
            for v in self.pheromone[u]:
                self.pheromone[u][v] *= (1 - self.rho)

        # Deposit pheromone for each ant's path
        for path, cost in ant_paths:
            if path is None or cost == float('inf'):
                continue
            deposit = 1.0 / cost
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                self.pheromone[u][v] += deposit
                self.pheromone[v][u] += deposit

    def solve(self, source, destination):
        """
        Execute the ACO algorithm to find the best routing path.

        Args:
            source:      Source router node
            destination: Destination router node

        Returns:
            best_path:  Best path found (list of nodes)
            best_cost:  Total latency of the best path
            history:    List of best cost found at each iteration (for convergence analysis)
        """
        if source not in self.graph.nodes:
            raise ValueError(f"Source node {source} not in the network")
        if destination not in self.graph.nodes:
            raise ValueError(f"Destination node {destination} not in the network")
        if source == destination:
            return [source], 0, [0] * self.num_iterations

        best_path = None
        best_cost = float('inf')
        history = []

        for iteration in range(self.num_iterations):
            ant_paths = []

            # Each ant constructs a path
            for _ in range(self.num_ants):
                path, cost = self._construct_path(source, destination)
                ant_paths.append((path, cost))

                # Update global best
                if cost < best_cost:
                    best_cost = cost
                    best_path = list(path)

            # Pheromone update
            self._update_pheromones(ant_paths)
            history.append(best_cost)

        return best_path, best_cost, history


# ============================================================
# Dijkstra's Algorithm (Deterministic Shortest Path)
# ============================================================
def dijkstra(graph, source, destination):
    """
    Find the shortest path using Dijkstra's algorithm with a min-heap.

    Args:
        graph:       Network Graph object
        source:      Source node
        destination: Destination node

    Returns:
        (path, cost): Shortest path and its total latency.
                      Returns (None, inf) if destination is unreachable.
    """
    if source not in graph.nodes:
        raise ValueError(f"Source node {source} not in the network")
    if destination not in graph.nodes:
        raise ValueError(f"Destination node {destination} not in the network")
    if source == destination:
        return [source], 0

    # Initialize distances and predecessors
    distances = {node: float('inf') for node in graph.nodes}
    predecessors = {node: None for node in graph.nodes}
    distances[source] = 0
    visited = set()
    priority_queue = [(0, source)]

    while priority_queue:
        current_dist, current = heapq.heappop(priority_queue)

        if current in visited:
            continue
        visited.add(current)

        if current == destination:
            break

        for neighbor, weight in graph.get_neighbors(current).items():
            if neighbor not in visited:
                new_dist = current_dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = current
                    heapq.heappush(priority_queue, (new_dist, neighbor))

    # Reconstruct shortest path
    if distances[destination] == float('inf'):
        return None, float('inf')

    path = []
    node = destination
    while node is not None:
        path.append(node)
        node = predecessors[node]
    path.reverse()

    return path, distances[destination]


# ============================================================
# Input File Parser
# ============================================================
def parse_input_file(filename):
    """
    Parse input file containing network topology cases.

    Expected format per case:
        Case X
        Nodes (Routers): 0, 1, 2, ...
        Edges (Links):
        u - v: weight
        ...
        Source: s
        Destination: d

    Returns:
        List of case dictionaries with keys:
        'name', 'graph', 'source', 'destination', 'nodes'
    """
    cases = []

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file '{filename}' not found.")
        sys.exit(1)
    except IOError as e:
        print(f"Error reading '{filename}': {e}")
        sys.exit(1)

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Detect start of a new case
        if line.lower().startswith('case'):
            case_name = line
            i += 1

            # Find and parse nodes line
            while i < len(lines) and 'node' not in lines[i].lower():
                i += 1
            if i >= len(lines):
                print(f"Error: Missing nodes definition for {case_name}")
                sys.exit(1)
            nodes_part = lines[i].split(':')[1].strip()
            nodes = [int(n.strip()) for n in nodes_part.split(',')]
            i += 1

            # Find edges header
            while i < len(lines) and 'edge' not in lines[i].lower():
                i += 1
            i += 1  # Skip the header line

            # Build graph and parse edge entries
            graph = Graph()
            for n in nodes:
                graph.add_node(n)

            while i < len(lines):
                edge_line = lines[i].strip()
                if (not edge_line or
                        edge_line.lower().startswith('source') or
                        edge_line.lower().startswith('case')):
                    break
                # Normalize en-dash / em-dash to hyphen
                edge_line = edge_line.replace('\u2013', '-').replace('\u2014', '-')
                parts = edge_line.split(':')
                if len(parts) == 2:
                    try:
                        weight = int(parts[1].strip())
                        node_parts = parts[0].split('-')
                        u = int(node_parts[0].strip())
                        v = int(node_parts[1].strip())
                        graph.add_edge(u, v, weight)
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Skipping malformed edge line: '{edge_line}' ({e})")
                i += 1

            # Parse source node
            while i < len(lines) and 'source' not in lines[i].lower():
                i += 1
            if i >= len(lines):
                print(f"Error: Missing source node for {case_name}")
                sys.exit(1)
            source = int(lines[i].split(':')[1].strip())
            i += 1

            # Parse destination node
            while i < len(lines) and 'destination' not in lines[i].lower():
                i += 1
            if i >= len(lines):
                print(f"Error: Missing destination node for {case_name}")
                sys.exit(1)
            destination = int(lines[i].split(':')[1].strip())
            i += 1

            cases.append({
                'name': case_name,
                'graph': graph,
                'source': source,
                'destination': destination,
                'nodes': nodes
            })
        else:
            i += 1

    if not cases:
        print("Error: No valid cases found in input file.")
        sys.exit(1)

    return cases


# ============================================================
# Utility Functions
# ============================================================
def format_path(path):
    """Format a path as 'n1 -> n2 -> ... -> nk'."""
    if path is None:
        return "No path found"
    return " -> ".join(str(node) for node in path)


def find_convergence_iteration(history):
    """Find the first iteration where the final best cost was achieved (1-indexed)."""
    if not history:
        return 0
    best = history[-1]
    for i, cost in enumerate(history):
        if cost == best:
            return i + 1
    return len(history)


# ============================================================
# Main Execution
# ============================================================
def main():
    """Main function: parse input, run algorithms, write output."""

    INPUT_FILE = "inputPS13.txt"
    OUTPUT_FILE = "outputPS13.txt"
    NUM_ITERATIONS = 100
    RANDOM_SEED = 42

    # ACO hyperparameter configurations (from assignment specification)
    scenarios = [
        {
            'name': 'Scenario 1',
            'num_ants': 10,
            'alpha': 1.0,
            'beta': 2.0,
            'rho': 0.5
        },
        {
            'name': 'Scenario 2',
            'num_ants': 10,
            'alpha': 2.5,
            'beta': 1.0,
            'rho': 0.3
        }
    ]

    # Parse input file
    cases = parse_input_file(INPUT_FILE)
    output_lines = []

    # ========== Part (a): PEAS Description ==========
    output_lines.append("=" * 65)
    output_lines.append("PEAS Description for the ACO Routing Agent")
    output_lines.append("=" * 65)
    output_lines.append("")
    output_lines.append("Performance Measure:")
    output_lines.append("  Minimize total transmission latency (sum of edge weights)")
    output_lines.append("  along the routing path from source to destination router.")
    output_lines.append("")
    output_lines.append("Environment:")
    output_lines.append("  A computer communication network modeled as a weighted")
    output_lines.append("  undirected graph. Nodes represent routers, edges represent")
    output_lines.append("  communication links with associated latency costs.")
    output_lines.append("  The environment is fully observable, static, discrete,")
    output_lines.append("  and deterministic.")
    output_lines.append("")
    output_lines.append("Actuators:")
    output_lines.append("  - Selection of the next router (hop) in the path")
    output_lines.append("  - Deposition of pheromone on traversed edges")
    output_lines.append("  - Pheromone evaporation mechanism")
    output_lines.append("")
    output_lines.append("Sensors:")
    output_lines.append("  - Current pheromone levels on all edges (tau)")
    output_lines.append("  - Edge latency values for heuristic computation (eta = 1/latency)")
    output_lines.append("  - Set of visited/unvisited nodes to avoid cycles")
    output_lines.append("  - Identity of current node and destination node")
    output_lines.append("")

    # ========== Part (b) & (c): Run algorithms and compare ==========
    for case in cases:
        graph = case['graph']
        source = case['source']
        destination = case['destination']
        nodes_str = ', '.join(str(n) for n in sorted(case['nodes']))

        output_lines.append("=" * 65)
        output_lines.append(f"{case['name']}")
        output_lines.append(f"Nodes (Routers): {nodes_str}")
        output_lines.append(f"Source: {source}, Destination: {destination}")
        output_lines.append("=" * 65)

        # --- Dijkstra's Algorithm ---
        try:
            dij_path, dij_cost = dijkstra(graph, source, destination)
        except ValueError as e:
            output_lines.append(f"\nDijkstra Error: {e}")
            dij_path, dij_cost = None, float('inf')

        output_lines.append("")
        output_lines.append("--- Dijkstra's Algorithm (Optimal Shortest Path) ---")
        output_lines.append(f"Best Path: {format_path(dij_path)}")
        output_lines.append(f"Minimum Latency: {dij_cost}")

        # --- ACO for each scenario ---
        scenario_results = []
        for sc in scenarios:
            random.seed(RANDOM_SEED)  # Reset seed for fair comparison

            try:
                aco = AntColonyOptimization(
                    graph=graph,
                    num_ants=sc['num_ants'],
                    alpha=sc['alpha'],
                    beta=sc['beta'],
                    rho=sc['rho'],
                    num_iterations=NUM_ITERATIONS
                )
                aco_path, aco_cost, history = aco.solve(source, destination)
            except ValueError as e:
                output_lines.append(f"\nACO {sc['name']} Error: {e}")
                continue

            conv_iter = find_convergence_iteration(history)
            matches = "Yes" if aco_cost == dij_cost else "No"

            scenario_results.append({
                'scenario': sc,
                'path': aco_path,
                'cost': aco_cost,
                'history': history,
                'conv_iter': conv_iter
            })

            output_lines.append("")
            output_lines.append(f"--- ACO {sc['name']} ---")
            output_lines.append(
                f"Parameters: alpha={sc['alpha']}, beta={sc['beta']}, "
                f"rho={sc['rho']}, ants={sc['num_ants']}, "
                f"iterations={NUM_ITERATIONS}"
            )
            output_lines.append(f"Best Path: {format_path(aco_path)}")
            output_lines.append(f"Minimum Latency: {aco_cost}")
            output_lines.append(f"Converged at Iteration: {conv_iter}")
            output_lines.append(f"Matches Dijkstra's Optimal: {matches}")

        # --- Part (c): Comparison Analysis ---
        if len(scenario_results) == 2:
            s1, s2 = scenario_results[0], scenario_results[1]

            output_lines.append("")
            output_lines.append("--- Comparison Analysis ---")

            # 1. Convergence speed
            output_lines.append("")
            output_lines.append("1. Convergence Speed:")
            output_lines.append(f"   Scenario 1 converged at iteration: {s1['conv_iter']}")
            output_lines.append(f"   Scenario 2 converged at iteration: {s2['conv_iter']}")
            if s1['conv_iter'] < s2['conv_iter']:
                output_lines.append(
                    "   -> Scenario 1 converged faster due to higher beta (2.0) which"
                )
                output_lines.append(
                    "      emphasizes heuristic information, guiding ants to shorter"
                )
                output_lines.append(
                    "      edges early in the search process."
                )
            elif s2['conv_iter'] < s1['conv_iter']:
                output_lines.append(
                    "   -> Scenario 2 converged faster due to higher alpha (2.5) which"
                )
                output_lines.append(
                    "      strongly reinforces discovered good paths through pheromone."
                )
            else:
                output_lines.append(
                    "   -> Both scenarios converged at the same iteration."
                )

            # 2. Comparison with optimal shortest path
            output_lines.append("")
            output_lines.append("2. Comparison with Dijkstra's Optimal Path:")
            output_lines.append(f"   Dijkstra's optimal cost: {dij_cost}")
            if dij_cost != float('inf'):
                dev1 = s1['cost'] - dij_cost
                dev2 = s2['cost'] - dij_cost
            else:
                dev1, dev2 = 'N/A', 'N/A'
            output_lines.append(
                f"   Scenario 1 ACO cost: {s1['cost']} (Deviation: {dev1})"
            )
            output_lines.append(
                f"   Scenario 2 ACO cost: {s2['cost']} (Deviation: {dev2})"
            )
            if s1['cost'] == dij_cost and s2['cost'] == dij_cost:
                output_lines.append(
                    "   -> Both ACO scenarios successfully found the optimal path."
                )
            elif s1['cost'] == dij_cost:
                output_lines.append(
                    "   -> Scenario 1 found the optimal path; Scenario 2 did not."
                )
            elif s2['cost'] == dij_cost:
                output_lines.append(
                    "   -> Scenario 2 found the optimal path; Scenario 1 did not."
                )
            else:
                output_lines.append(
                    "   -> Neither scenario found the exact optimal path,"
                )
                output_lines.append(
                    "      which is expected for ACO as a metaheuristic."
                )

            # 3. Effect of parameter settings
            output_lines.append("")
            output_lines.append("3. Effect of Parameters (alpha, beta, rho) on Solution Quality:")
            output_lines.append("")
            output_lines.append(
                "   Scenario 1 (alpha=1.0, beta=2.0, rho=0.5):"
            )
            output_lines.append(
                "     - Higher beta (2.0) places more emphasis on the heuristic"
            )
            output_lines.append(
                "       (inverse latency), guiding ants toward shorter edges."
            )
            output_lines.append(
                "     - Moderate alpha (1.0) balances pheromone influence."
            )
            output_lines.append(
                "     - Higher evaporation rate (rho=0.5) promotes exploration by"
            )
            output_lines.append(
                "       reducing old pheromone faster, preventing premature convergence."
            )
            output_lines.append("")
            output_lines.append(
                "   Scenario 2 (alpha=2.5, beta=1.0, rho=0.3):"
            )
            output_lines.append(
                "     - Higher alpha (2.5) heavily favors pheromone-rich paths,"
            )
            output_lines.append(
                "       leading to stronger exploitation of discovered routes."
            )
            output_lines.append(
                "     - Lower beta (1.0) reduces the influence of edge latency"
            )
            output_lines.append(
                "       heuristic, making path selection more pheromone-dependent."
            )
            output_lines.append(
                "     - Lower evaporation rate (rho=0.3) retains pheromone longer,"
            )
            output_lines.append(
                "       reinforcing previously found paths but risking premature"
            )
            output_lines.append(
                "       convergence to suboptimal solutions."
            )

        output_lines.append("")

    # Write output to file
    try:
        with open(OUTPUT_FILE, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"Output successfully written to {OUTPUT_FILE}")
    except IOError as e:
        print(f"Error writing output file '{OUTPUT_FILE}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
