import sys

import pandas as pd
import numpy as np
import networkx as nx
from itertools import product
from collections import defaultdict
from scipy.special import gammaln

from itertools import combinations
from scipy.stats import chi2_contingency



import pandas as pd
from scipy.stats import fisher_exact
# Fisher's exact test (same as before)
def fisher_test(data, var1, var2, condition_set):
    if condition_set:
        if isinstance(condition_set, tuple):
            condition_set = list(condition_set)  # Convert tuple to list
        elif isinstance(condition_set, str):
            condition_set = [condition_set]

        grouped_data = data.groupby(condition_set)
        p_values = []
        for _, group in grouped_data:
            contingency_table = pd.crosstab(group[var1], group[var2])
            if contingency_table.shape == (2, 2):
                odds_ratio, p = fisher_exact(contingency_table)
                p_values.append(p)
            elif contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                chi2, p, dof, ex = chi2_contingency(contingency_table)
                p_values.append(p)
            else:
                continue
        
        if p_values:
            avg_p_value = sum(p_values) / len(p_values)
        else:
            avg_p_value = 1.0  # Default high p-value for independence
        return avg_p_value
    else:
        contingency_table = pd.crosstab(data[var1], data[var2])
        if contingency_table.shape == (2, 2):
            odds_ratio, p_value = fisher_exact(contingency_table)
        else:
            chi2, p_value, dof, ex = chi2_contingency(contingency_table)
        return p_value

# Grow-Shrink algorithm implementation
def gs_algorithm(data, significance_level=0.05):
    """
    GS algorithm to learn the structure of a Bayesian network from data.
    
    Parameters:
    - data: The dataset as a pandas DataFrame.
    - significance_level: The significance level for the conditional independence test.
    
    Returns:
    - dag: The learned directed acyclic graph (DAG) as a NetworkX DiGraph.
    """
    nodes = data.columns
    dag = nx.DiGraph()

    for node in nodes:
        parents = grow_phase(data, node, significance_level)
        parents = shrink_phase(data, node, parents, significance_level)
        for parent in parents:
            # Check if both nodes exist in the graph
            if parent in dag.nodes and node in dag.nodes:
                # Check if adding the edge will create a cycle
                if not nx.has_path(dag, node, parent):  # Only add edge if it does not create a cycle
                    dag.add_edge(parent, node)
            else:
                # If either node is not in the graph, safely add the edge
                dag.add_edge(parent, node)

    return dag

def grow_phase(data, target, significance_level):
    """
    Grow phase: Find the set of parents of the target node.
    
    Parameters:
    - data: The dataset as a pandas DataFrame.
    - target: The target node to find parents for.
    - significance_level: The significance level for conditional independence test.
    
    Returns:
    - A list of potential parents for the target node.
    """
    potential_parents = []
    nodes = set(data.columns) - {target}

    # Grow phase: Add variables that are dependent on the target
    for var in nodes:
        if fisher_test(data, target, var, []) <= significance_level:
            potential_parents.append(var)
    
    return potential_parents

def shrink_phase(data, target, potential_parents, significance_level):
    """
    Shrink phase: Remove parents that are conditionally independent given the other parents.
    
    Parameters:
    - data: The dataset as a pandas DataFrame.
    - target: The target node to find parents for.
    - potential_parents: List of potential parents found in the grow phase.
    - significance_level: The significance level for conditional independence test.
    
    Returns:
    - A list of valid parents for the target node.
    """
    parents = potential_parents.copy()

    # Shrink phase: Remove variables that are conditionally independent
    for parent in potential_parents:
        other_parents = set(parents) - {parent}
        if fisher_test(data, target, parent, list(other_parents)) > significance_level:
            parents.remove(parent)

    return parents
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
    graph = gs_algorithm(data)  # Use GS instead of PC
    score = log_bayesian_score(data, graph)
    print(graph)
    write_gph(graph, data, outfile)
    print(score)



def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
