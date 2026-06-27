"""
Multiple ways to address overfitting:
    * Apply a minimum number of examples per leaf
    * Apply max depth to limit growth of tree
    * Prune decision tree based on validation set (remove nodes that improve val metric)
"""

from decision_tree.classification_tree.core import LeafNode


def cost_complexity_metric(tree, samples, target_attribute):
    pass

def cost_complexity_pruning(samples, tree, alpha, k=5, val_ratio=0.1):
    """
    Post-pruning based on cost-complexity pruning.
    Prune nodes that reduce the cost-complexity metric on a validation set.

    Cost-complexity metric = empirical risk + alpha * number of leaf nodes

    Args:
        samples: dataset to split into training and validation sets
        tree: decision tree to prune
        alpha: complexity parameter - the higher the alpha, the simpler the pruned tree
        val_ratio: ratio of samples to use as validation set
    Returns:
        pruned decision tree
    """
    # Split samples into training and validation sets using k-fold cross-validation
    fold_size = len(samples) // k
    for i in range(k):
        val_samples = samples[i * fold_size:(i + 1) * fold_size]
        train_samples = samples[:i * fold_size] + samples[(i + 1) * fold_size:]



    def count_leaves(tree):
        if isinstance(tree, LeafNode):
            return 1
        return sum(count_leaves(subtree) for subtree in tree.subtrees.values())

    def prune(tree):
        pass
