import sys
import pandas as pd
import numpy as np
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from scipy.special import gammaln
import random


def mcmc(data, initial_graph, alpha=1.0, max_iterations=10000):
    """
    MCMC to find the best graph structure that maximizes the log Bayesian score.
    """
    current_graph = initial_graph.copy()
    current_score = log_bayesian_score(data, current_graph, alpha)
    best_graph = current_graph.copy()
    best_score = current_score

    for _ in range(max_iterations):
        if _ % 100 == 0:
            print(f"Iteration {_}")
        # Generate a neighboring graph by modifying the current one
        neighbor_graph = generate_neighbor(current_graph)
        neighbor_score = log_bayesian_score(data, neighbor_graph, alpha)

        # Metropolis-Hastings acceptance criteria
        if neighbor_score > current_score:
            current_graph = neighbor_graph
            current_score = neighbor_score
        else:
            acceptance_prob = np.exp(neighbor_score - current_score)
            if random.uniform(0, 1) < acceptance_prob:
                current_graph = neighbor_graph
                current_score = neighbor_score

        if current_score > best_score:
            best_graph = current_graph.copy()
            best_score = current_score

    return best_graph, best_score


def generate_neighbor(graph):
    """
    Generate a neighboring DAG by randomly adding, removing, or reversing an edge.
    Ensures that the resulting graph is acyclic.
    """
    neighbor = graph.copy()
    nodes = list(graph.nodes)

    operation = random.choice(["add", "remove", "reverse"])

    if operation == "add":
        u, v = random.sample(nodes, 2)
        if not neighbor.has_edge(u, v) and not neighbor.has_edge(v, u):
            neighbor.add_edge(u, v)
            if not nx.is_directed_acyclic_graph(neighbor):
                neighbor.remove_edge(u, v)  # Undo if it creates a cycle
    elif operation == "remove" and neighbor.edges:
        u, v = random.choice(list(neighbor.edges))
        neighbor.remove_edge(u, v)
    elif operation == "reverse" and neighbor.edges:
        u, v = random.choice(list(neighbor.edges))
        neighbor.remove_edge(u, v)
        neighbor.add_edge(v, u)
        if not nx.is_directed_acyclic_graph(neighbor):
            neighbor.remove_edge(v, u)
            neighbor.add_edge(u, v)

    return neighbor


def bayesian_score_component(M, alpha):
    """
    Compute the Bayesian score component for a single variable.
    """
    alpha_sum = np.sum(alpha, axis=1)
    M_sum = np.sum(M, axis=1)

    p = np.sum(gammaln(alpha + M))
    p -= np.sum(gammaln(alpha))
    p += np.sum(gammaln(alpha_sum))
    p -= np.sum(gammaln(alpha_sum + M_sum))

    return p


def log_bayesian_score(data: pd.DataFrame, dag: nx.DiGraph, alpha: float = 1.0) -> float:
    D = data.to_numpy()
    variables = [Variable(data[col].max()) for col in data.columns]
    node_to_index = {node: idx for idx, node in enumerate(data.columns)}

    M = statistics(variables, dag, D, node_to_index)
    alpha_matrix = prior(variables, dag, node_to_index)

    with ThreadPoolExecutor() as executor:
        scores = executor.map(lambda i: bayesian_score_component(M[i], alpha_matrix[i]), range(len(variables)))

    return sum(scores)


class Variable:
    def __init__(self, num_states):
        self.r = num_states


def sub2ind(siz, x):
    k = np.concatenate(([1], np.cumprod(siz[:-1])))
    return np.dot(k, np.array(x))


def statistics(vars, G, D, node_to_index):
    D = D - 1  # To account for the 1-indexing
    m, n = D.shape
    r = [vars[i].r for i in range(n)]

    q = [np.prod([r[node_to_index[j]] for j in G.predecessors(node)]) if len(list(G.predecessors(node))) > 0 else 1 for node in G.nodes]

    M = [np.zeros((q[i], r[i])) for i in range(n)]

    for o in D:
        for i, node in enumerate(G.nodes):
            k = o[node_to_index[node]]
            if k >= r[i]:
                print(i, node, k, r[i])
                raise ValueError(f"State value {k} for variable {node} exceeds the expected number of states ({r[i]}).")

            parents = list(G.predecessors(node))

            if parents:
                parent_states = [o[node_to_index[j]] for j in parents]
                parent_sizes = [r[node_to_index[j]] for j in parents]
                j = sub2ind(parent_sizes, parent_states)
            else:
                j = 0

            M[i][int(j), int(k)] += 1

    return M


def prior(vars, G, node_to_index):
    n = len(vars)
    r = [vars[i].r for i in range(n)]
    prior_matrices = []

    for i, node in enumerate(G.nodes):
        parents = list(G.predecessors(node))

        if parents:
            q_i = np.prod([r[node_to_index[j]] for j in parents])
        else:
            q_i = 1

        prior_matrix = np.ones((q_i, r[i]))
        prior_matrices.append(prior_matrix)

    return prior_matrices


def write_gph(dag, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(edge[0], edge[1]))


def compute(infile, outfile):
    data = pd.read_csv(infile)
    print(data.shape)
    nodes = data.columns
    initial_graph = nx.DiGraph()
    initial_graph.add_nodes_from(nodes)

    best_graph, best_score = mcmc(data, initial_graph, alpha=1.0, max_iterations=5000)

    write_gph(best_graph, outfile)
    print(best_score)


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
