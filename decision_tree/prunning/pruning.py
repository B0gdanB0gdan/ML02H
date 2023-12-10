"""
Multiple ways to address overfitting:
    * Apply a minimum number of examples per leaf
    * Apply max depth to limit growth of tree
    * Prune decision tree based on validation set (remove nodes that improve val metric)
"""


def cost_complexity_pruning(samples, tree, alpha, val_ratio=0.1):
    pass
