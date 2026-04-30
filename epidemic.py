import random
import numpy as np

def si_epidemic(graph, beta, initial_infected_count=1, max_iterations=None, max_infected_frac=1):
    """
    Simulate a Susceptible-Infected (SI) epidemic model on a given graph.
    In the SI model, susceptible nodes become infected through contact with infected
    neighbors and remain infected permanently (no recovery). The simulation proceeds
    in discrete time steps until a stopping condition is met.

    Parameters
    ----------
    graph : NetworkX graph
        The contact network on which the epidemic spreads. Nodes represent individuals
        and edges represent contact relationships through which infection can spread.
    beta : float
        Transmission probability per contact, in range [0, 1]. At each iteration, an
        infected node attempts to infect each susceptible neighbor independently with
        this probability.
    initial_infected_count : int, optional
        Number of nodes to infect at the start of the simulation. These nodes are
        selected uniformly at random from all nodes in the graph. Default is 1.
    max_iterations : int or None, optional
        Maximum number of simulation iterations. If None, the simulation continues
        until no new infections occur or max_infected_frac is reached. Default is None.
    max_infected_frac : float or None, optional
        Maximum fraction of nodes that can become infected, in range [0, 1]. The 
        simulation stops when this threshold is reached or exceeded. If None, no
        limit is imposed. Default is 1 (all nodes can be infected).
    
    Returns
    -------
    list
        List of all infected node IDs at the end of the simulation, including both
        initially infected nodes and those infected during the simulation.
    
    Notes
    -----
    - The simulation terminates when any of the following occurs:
        1. max_iterations is reached (if specified)
        2. The number of infected nodes reaches max_infected_frac threshold
    - Infected nodes are shuffled each iteration to prevent ordering bias in the
      infection process.
    - The max_infected limit is enforced strictly during each iteration to prevent
      overshooting the threshold.
    """
    
    # Randomly select the initial infected nodes from the graph
    initial_infected = random.sample(list(graph.nodes()), k=initial_infected_count) # Randomly select the initial infected nodes from the graph
    infected = set(initial_infected)
    iteration = 0

    # Calculate the maximum number of infected nodes based on the specified fraction
    max_infected = None if max_infected_frac is None else int(max_infected_frac * len(graph))

    while True:
        # Stopping condition 1: Maximum iterations reached
        if max_iterations is not None and iteration >= max_iterations:
            break
        # Stopping condition 2: Maximum infected threshold reached
        if len(infected) >= max_infected:
            break

        new_infected = set()        
        infected_list = list(infected)
        random.shuffle(infected_list) # Shuffle to avoid bias from node ordering

        for node in infected_list:
            # Iterate over neighbors of the infected node that are not already infected
            for neighbor in set(graph.neighbors(node)) - infected:
                # Ensure maximum number of infected nodes is not exceeded
                if max_infected is not None and len(infected) + len(new_infected) >= max_infected:
                    break
                # With probability 'beta', attempt to infect the neighbor
                if random.random() < beta:
                    new_infected.add(neighbor) # Add the neighbor to the new infected set

        infected.update(new_infected) # Update the infected set with the newly infected nodes
        iteration += 1

    return list(infected)


def observed_infected(infected_nodes, observation_probability):
    """
    Simulate partial observation of infected nodes in an epidemic.    
    Models the realistic scenario where not all infected individuals are detected.
    Each infected node is independently observed with the specified probability.

    Parameters
    ----------
    infected_nodes : list
        List of infected node IDs.
    observation_probability : float
        Probability of observing each infected node, in range [0, 1]. A value of 1.0
        means perfect observation (all infected nodes detected), while 0.0 means no
        observation.
    
    Returns
    -------
    list
        Subset of infected_nodes that were successfully observed.
        Returns an empty list if no nodes are observed.
    
    """
    # Create an array where each element equals the observation probability
    observation_probs = np.full(len(infected_nodes), observation_probability)
    # Generate random uniform samples between 0 and 1 for each infected node
    random_draws = np.random.uniform(size=len(infected_nodes))
    # Create binary mask: 1 if the infected node is observed, 0 otherwise
    observation_mask = np.multiply(random_draws < observation_probs, 1)

    # Filter infected nodes using the observation mask
    observed_nodes = [x for x, mask in zip(infected_nodes, observation_mask) if mask==1]
    
    return observed_nodes