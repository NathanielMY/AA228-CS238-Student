import sys

import pandas as pd
import numpy as np
import networkx as nx
from itertools import product
from collections import defaultdict
from scipy.special import gammaln

from itertools import combinations

import random
import networkx as nx
import numpy as np

def simulated_annealing(data, initial_graph, alpha=1.0, initial_temp=1000, min_temp=1, cooling_rate=0.95, max_iterations=1000):
    """
    Simulated annealing to find the best graph structure that maximizes the log Bayesian score.

    Parameters:
    - data: The dataset as a pandas DataFrame.
    - initial_graph: The initial DAG (NetworkX DiGraph) to start with.
    - alpha: The Dirichlet prior hyperparameter.
    - initial_temp: Initial temperature for annealing.
    - min_temp: The minimum temperature at which to stop annealing.
    - cooling_rate: The rate at which the temperature decreases (0 < cooling_rate < 1).
    - max_iterations: Maximum number of iterations for annealing.

    Returns:
    - best_graph: The DAG with the best log Bayesian score found.
    - best_score: The corresponding log Bayesian score.
    """
    current_graph = initial_graph.copy()  # Start with the initial graph
    current_score = log_bayesian_score(data, current_graph, alpha)  # Compute its score
    best_graph = current_graph.copy()
    best_score = current_score

    temp = initial_temp

    while temp > min_temp and max_iterations > 0:
        # Generate a neighboring graph by modifying the current one
        neighbor_graph = generate_neighbor(current_graph)
        neighbor_score = log_bayesian_score(data, neighbor_graph, alpha)

        # If the neighbor has a better score, accept it
        if neighbor_score > current_score:
            current_graph = neighbor_graph
            current_score = neighbor_score

            # Update best graph if this is the best so far
            if current_score > best_score:
                best_graph = current_graph
                best_score = current_score
        else:
            # If the neighbor has a worse score, accept it with some probability
            acceptance_prob = np.exp((neighbor_score - current_score) / temp)
            if random.uniform(0, 1) < acceptance_prob:
                current_graph = neighbor_graph
                current_score = neighbor_score

        # Cool down the temperature
        temp *= cooling_rate
        max_iterations -= 1

    return best_graph, best_score

def generate_neighbor(graph):
    """
    Generate a neighboring DAG by randomly adding, removing, or reversing an edge.
    Ensures that the resulting graph is acyclic.
    """
    neighbor = graph.copy()
    nodes = list(graph.nodes)

    # Randomly decide whether to add, remove, or reverse an edge
    operation = random.choice(["add", "remove", "reverse"])

    if operation == "add":
        # Add a random edge if it does not already exist and does not create a cycle
        u, v = random.sample(nodes, 2)
        if not neighbor.has_edge(u, v) and not neighbor.has_edge(v, u):
            neighbor.add_edge(u, v)
            if not nx.is_directed_acyclic_graph(neighbor):
                neighbor.remove_edge(u, v)  # Undo if it creates a cycle
    elif operation == "remove":
        # Remove a random edge
        if neighbor.edges:
            u, v = random.choice(list(neighbor.edges))
            neighbor.remove_edge(u, v)
    elif operation == "reverse":
        # Reverse a random edge
        if neighbor.edges:
            u, v = random.choice(list(neighbor.edges))
            neighbor.remove_edge(u, v)
            neighbor.add_edge(v, u)
            if not nx.is_directed_acyclic_graph(neighbor):
                neighbor.remove_edge(v, u)  # Undo if it creates a cycle
                neighbor.add_edge(u, v)

    return neighbor


def log_bayesian_score(data: pd.DataFrame, dag: nx.DiGraph, alpha: float = 1.0) -> float:
    """
    Compute the log Bayesian score (log BDe score) of a Bayesian Network (DAG) given a dataset.

    Parameters:
    - data: A pandas DataFrame representing the dataset, where each column is a variable.
    - dag: A NetworkX DiGraph representing the structure of the Bayesian Network.
    - alpha: The Dirichlet prior hyperparameter. Default is 1 (uniform Dirichlet prior).

    Returns:
    - The log Bayesian score of the given DAG and dataset.
    """
    # Check if all nodes in the DAG are in the dataset
    if not set(dag.nodes()).issubset(set(data.columns)):
        raise ValueError("All nodes in the DAG must be columns in the dataset.")

    log_score = 0.0  # Total log Bayesian score

    # Iterate over each node in the DAG
    for node in dag.nodes:
        parents = list(dag.predecessors(node))  # Get the parents of the current node

        # Get the counts of the joint configurations of the node and its parents
        if parents:
            # Count occurrences of each unique parent configuration and child state
            grouped_counts = data.groupby(parents + [node]).size().unstack(fill_value=0)
            parent_counts = data.groupby(parents).size()
        else:
            # No parents, so just count occurrences of the node's states
            grouped_counts = data.groupby([node]).size()
            grouped_counts = grouped_counts.reset_index(name='count')  # Convert to DataFrame with 'count'
            parent_counts = pd.Series([len(data)], index=[0])  # No parents case

        # Get the number of possible values for the node and its parents
        r_i = len(grouped_counts)  # Number of states for node
        q_i = len(parent_counts)   # Number of unique parent configurations

        # For each parent configuration (if any), calculate the local log marginal likelihood
        for j in range(q_i):
            # Count N_{ijk}, the number of times node i = k, given parent configuration j
            if parents:
                N_ij = grouped_counts.iloc[j].values
            else:
                N_ij = grouped_counts['count'].values  # Use 'count' column when no parents

            # Calculate log marginal likelihood for the parent configuration
            log_term_1 = gammaln(alpha) - gammaln(alpha + parent_counts.iloc[j])

            # Summation over all k states of node
            log_term_2 = np.sum(gammaln(N_ij + alpha / r_i) - gammaln(alpha / r_i))

            log_score += log_term_1 + log_term_2

    return log_score



def generate_idx2names(data):
    """
    Generates a dictionary mapping indices to variable names based on the DataFrame's columns.
    
    Parameters:
    - data: A pandas DataFrame where each column is a variable (node).
    
    Returns:
    - idx2names: A dictionary mapping indices to column names.
    """
    return {name: idx for idx, name in enumerate(data.columns)}


def write_gph(dag, data, filename):
    idx2names = generate_idx2names(data)
    print(idx2names)
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(edge[0], edge[1]))




def compute(infile, outfile):
    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING

    #steps - read the file in to get data
    #write a function to compute bayesian score of a graph
    #have some way to iterate through graphs and maybe punish the ones that have too many edges
    #store the graph with the highest score

    #infile is the csv file

    data = pd.read_csv(infile)
    print(data.shape)
    nodes = data.columns
    initial_graph = nx.DiGraph()  # Start with an empty graph

    # Add nodes to the graph
    initial_graph.add_nodes_from(nodes)

    # Run simulated annealing to find the best graph
    best_graph, best_score = simulated_annealing(data, initial_graph, alpha=1.0, initial_temp=50000, min_temp=1, cooling_rate=0.99, max_iterations=10000)
    write_gph(best_graph, data, outfile)
    print(best_score)



def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
