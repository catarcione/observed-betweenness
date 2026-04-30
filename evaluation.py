from sklearn.metrics import roc_auc_score, roc_curve

def auc_score(infected_nodes, centrality_measures, observed_nodes):
    """
    Calculate the AUC score and ROC curve, given the set of infected nodes,
    the network centrality measure used for ranking, and a list of observed 
    nodes to be excluded from the calculation.

    Parameters
    ----------
    infected_nodes : list or set
        Collection of all node IDs that are truly infected in the epidemic,
        including both observed and unobserved infections. This represents the
        ground truth for evaluation.
    centrality_measures : dict
        Dictionary mapping node IDs to their centrality scores (float). Higher
        scores should indicate nodes more likely to be infected.
    observed_nodes : list or set
        Collection of node IDs that are known to be infected (observed cases).
        These nodes are excluded from evaluation since their infection status is
        already known and shouldn't be predicted.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics with the following keys:
        
        'auc' : float
            Area Under the ROC Curve score, in range [0, 1]:
            - 1.0: Perfect ranking (all infected nodes ranked above all susceptible)
            - 0.5: Random ranking (no discriminative power)
            - < 0.5: Worse than random (inverted ranking)
        
        'roc' : tuple of three numpy.ndarray
            ROC curve components for visualization:
            - fpr: False Positive Rate values (1D array)
            - tpr: True Positive Rate values (1D array)
            - thresholds: Decision threshold values (1D array)
            Can be unpacked as: fpr, tpr, thresholds = result['roc']

    Notes
    -----
    - If all evaluation nodes have the same centrality score, AUC cannot be
      computed and scikit-learn will raise a ValueError.
    - The ROC curve plots TPR (sensitivity) vs FPR (1 - specificity) at various
      classification thresholds on the centrality scores.
    - This evaluation assumes binary classification: infected (1) vs susceptible (0).
    """
    
    # Filter out observed nodes from evaluation
    # Only evaluate on nodes whose infection status is unknown/unobserved
    evaluation_nodes = [node for node in centrality_measures if node not in observed_nodes]

    # Build ground truth labels and predicted scores for evaluation nodes
    y_true = [] # Binary labels: 1 if infected, 0 if susceptible
    y_scores = [] # Centrality scores used for ranking

    for node in evaluation_nodes:
        y_true.append(1 if node in infected_nodes else 0) # Label: 1 if the node is truly infected, 0 otherwise
        y_scores.append(centrality_measures[node]) # Score: the centrality measure value for this node

    # Compute the Area Under the ROC Curve
    # Measures how well centrality scores separate infected from susceptible nodes
    auc = roc_auc_score(y_true, y_scores)
    # Compute the complete ROC curve (FPR, TPR, thresholds)
    # Useful for visualization and threshold selection
    curve = roc_curve(y_true, y_scores)

    return {'auc': auc, 'roc': curve}