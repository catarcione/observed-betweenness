import networkx as nx

def degree(graph):
    """
    Compute normalized degree centrality for all nodes in the graph.
    Calculates the fraction of nodes that each node is connected to, providing
    a measure of how well-connected each node is relative to the total network
    size.

    Parameters
    ----------
    graph : NetworkX graph
        The contact network on which the epidemic spreads. Nodes represent individuals
        and edges represent contact relationships through which infection can spread.
    
    Returns
    -------
    dict
        Dictionary mapping each node ID to its normalized degree centrality score.
        Values are in range [0, 1], where:
        - 0 indicates an isolated node (no connections)
        - 1 indicates a node connected to all other nodes in the graph
        
    Notes
    -----
    - In epidemic contexts, higher degree centrality suggests nodes that may be
      more likely to spread or receive infection due to more contacts.
    """
    
    return nx.degree_centrality(graph)


def contact(graph, observed_nodes):
    """
    Compute the fraction of neighbors that are observed infected for each node.
    For each node in the graph, calculates what proportion of its immediate neighbors
    are among the observed infected nodes. This metric identifies nodes that have
    high exposure to observed infection, which may indicate higher likelihood of being
    an infection source or being infected themselves.
    
    Parameters
    ----------
    graph : NetworkX graph
        The contact network on which the epidemic spreads. Nodes represent individuals
        and edges represent contact relationships through which infection can spread.
    observed_nodes : set or list
        Collection of node IDs that are known to be infected. Typically a subset
        of all infected nodes, representing those that were successfully detected
        or observed.
    
    Returns
    -------
    dict
        Dictionary mapping each node ID to its contact score (float in range [0, 1]):
        - 0.0: None of the node's neighbors are observed infected
        - 1.0: All of the node's neighbors are observed infected
    """
    
    result = {}
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        # Count how many neighbors are in the observed infected set
        num_infected_neighbors = sum(1 for neighbor in neighbors if neighbor in observed_nodes)
        # Compute fraction, handling isolated nodes (no neighbors) as 0.0
        frac_infected_neighbors = num_infected_neighbors/len(neighbors) if neighbors else 0.0
        result[node] = frac_infected_neighbors
    
    return result


def observed_betweenness(graph, observed_nodes):
    """
    Compute betweenness centrality using only paths between observed infected nodes.
    Calculates a restricted form of betweenness centrality where only shortest paths
    that start and end at observed infected nodes are considered. This identifies
    nodes that act as bridges or connectors within the observed infection subnetwork,
    which may indicate nodes that are central to the epidemic spread pattern.
    
    Parameters
    ----------
    graph : NetworkX graph
        The contact network on which the epidemic spreads. Nodes represent individuals
        and edges represent contact relationships through which infection can spread.
    observed_nodes : set or list
        Collection of node IDs that are known to be infected. Typically a subset
        of all infected nodes, representing those that were successfully detected
        or observed.
        The betweenness calculation only considers shortest paths where both 
        the source and target are in this set.

    Returns
    -------
    dict
        Dictionary mapping each node ID to its observed betweenness centrality score.
        Values are normalized floats:
        - 0.0: The node lies on no shortest paths between observed nodes
    """

    return nx.betweenness_centrality_subset(graph, sources=observed_nodes, targets=observed_nodes)
