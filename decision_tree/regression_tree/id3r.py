"""
    For each attribute: choose value as threshold based on biggest variance reduction (it also minimizes the MSE)
"""

from collections import defaultdict
from typing import Any, NamedTuple, Union, Dict

import numpy as np


class LeafNode(NamedTuple):
    mean_value: Any


class DecisionNode(NamedTuple):
    attribute: str
    threshold: int
    subtrees: dict


DecisionTree = Union[LeafNode, DecisionNode]


def partition_by(samples, attribute, threshold) -> Dict:
    partition = defaultdict(list)
    for sample in samples:
        if getattr(sample, attribute) <= threshold:
            partition[f"{attribute}<={threshold}"].append(sample)
        else:
            partition[f"{attribute}>{threshold}"].append(sample)
    return partition


def predict(tree, sample):
    if isinstance(tree, LeafNode):
        return tree.mean_value

    attr_val = getattr(sample, tree.attribute)

    subtree = tree.subtrees[
        f"{tree.attribute}<={tree.threshold}" if attr_val <= tree.threshold else f"{tree.attribute}>{tree.threshold}"]
    return predict(subtree, sample)


def variance_reduction(samples, attribute, target_attribute):
    """
        Metric of purity.
    """
    parent_variance = np.var([getattr(sample, target_attribute) for sample in samples])
    thresholds = np.unique([getattr(sample, attribute) for sample in samples])
    reductions = []
    for threshold in thresholds:
        partition = partition_by(samples, attribute, threshold).values()
        children_var = sum([len(subset) / len(samples) *
                            np.var([getattr(sample, target_attribute) for sample in subset]) for subset in partition])
        reductions.append((parent_variance - children_var, threshold))
    # min performed on reduction but still return tuple
    return max(reductions, key=lambda x: x[0])


def id3r(samples, attributes, target_attribute, min_samples=5, max_depth=3):
    return _id3r(samples, attributes, target_attribute, depth=0, min_samples=min_samples, max_depth=max_depth)


def _id3r(samples, attributes, target_attribute, depth, min_samples=5, max_depth=3):
    """Iterative Dichotomiser 3 for regression.
    Greedy algorithm: chooses the most immediate best option even if there might be a better tree with a worse-looking
    first move.
    Algorithm:
        *
    """

    targets = [getattr(sample, target_attribute) for sample in samples]

    # no variance
    c1 = np.var(targets) == 0

    # If no split attributes left, return the majority label
    c2 = not attributes

    # If remaining samples are less than min_examples
    c3 = len(samples) < min_samples

    # If depth reached max_depth
    c4 = depth == max_depth

    if c1 or c2 or c3 or c4:
        return LeafNode(np.round(np.mean(targets), 3))

    # mapping from attributes to (reduction, threshold) pairs
    var_red_map = {attr: variance_reduction(samples=samples, attribute=attr, target_attribute=target_attribute)
                   for attr in attributes}

    # choose the attribute that resulted in the biggest reduction in variance of the subtrees
    # in terms of the target attribute
    best_attribute = max(var_red_map, key=lambda k: var_red_map[k][0])
    _, threshold = var_red_map[best_attribute]

    partitions = partition_by(samples, best_attribute, threshold)

    new_attributes = [a for a in attributes if a != best_attribute]
    # Recursively build the subtrees
    subtrees = {attribute_value: _id3r(subset,
                                       new_attributes,
                                       target_attribute,
                                       depth=depth + 1,
                                       min_samples=min_samples,
                                       max_depth=max_depth)
                for attribute_value, subset in partitions.items()}
    return DecisionNode(best_attribute, threshold, subtrees)
