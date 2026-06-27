from collections import Counter, defaultdict
from typing import List, Any, NamedTuple, Union, Dict

import numpy as np
import matplotlib.pyplot as plt


class LeafNode(NamedTuple):
    value: Any


class DecisionNode(NamedTuple):
    attribute: str
    subtrees: dict
    default_value: Any = None


DecisionTree = Union[LeafNode, DecisionNode]


def entropy(probs):
    return sum(-p * np.log2(p) for p in probs if p > 0)


def entropy_details():
    """
    Small entropy when pi is close to 0 or 1 (most of the data is in one class).
    Large entropy when pi is not close to 1 (data is spread across classes).
    """

    x = np.linspace(0, 1, 100)
    y = [entropy([xi]) for xi in x]

    print(f"Max value: {max(y)}")
    print(f"Value in 0.5: {entropy([0.5])}")
    assert entropy([0.5, 0.5]) == 1
    assert entropy([0.01, 0.99]) < entropy([0.3, 0.7])

    plt.plot(x, y)
    plt.title("Entropy function")
    plt.xlabel("p")
    plt.ylabel("-p*log2(p)")
    plt.grid()
    plt.show()


def labels_to_probs(labels):
    count = Counter(labels).values()
    return [c / len(labels) for c in count]


def data_entropy(labels):
    return entropy(labels_to_probs(labels))


def partition_entropy(subsets: List[List[Any]]):
    """
    Partition has low entropy if the subsets of the partition have low entropy themselves.
    E.g. one attribute splits the data into 2 subsets where one subset belongs to c1 and the other subset to c2).
    Such subsets have high information (they have low entropy i.e. are highly certain) about what the prediction should be.
    Partitioning by an attribute with many different values will result in subsets with low entropy but with low generalization
    capability. Partitioning by CNP => one person subsets => each person has exactly one label. What happens if we test the model
    on a CNP key different than any value seen during training? The model overfitted to the CNP values seen during training.
    """
    n = sum([len(s) for s in subsets])
    return sum([len(s) / n * data_entropy(s) for s in subsets])


def partition_by(samples, attribute) -> Dict:
    """
    Partitions samples by the given attribute.
    """
    partition = defaultdict(list)
    for sample in samples:
        partition[getattr(sample, attribute)].append(sample)
    return partition


def entropy_of_partition_by(samples, attribute, target_attribute):
    """
    Computes the entropy of the partitioning of the samples by the given attribute.
    """
    partition = partition_by(samples, attribute)
    labels = [[getattr(sample, target_attribute) for sample in subset]
              for subset in partition.values()]
    return partition_entropy(labels)


def information_gain(samples, attribute, target_attribute):
    """
    Information gain measures how much a feature reduces uncertainty about a target variable,
    calculated by subtracting the entropy after a split from the entropy before the split.
    """
    before_split = data_entropy([getattr(sample, target_attribute) for sample in samples])
    after_split = entropy_of_partition_by(samples, attribute, target_attribute)
    return before_split - after_split


def gain_ratio(samples, attribute, target_attribute):
    """
    Gain ratio adjusts information gain by taking into account the intrinsic information of a split,
    which measures how broadly and uniformly the data is divided by the attribute.
    It is calculated by dividing the information gain by the intrinsic information.
    """
    info_gain = information_gain(samples, attribute, target_attribute)
    partition = partition_by(samples, attribute)
    n = len(samples)
    intrinsic_info = -sum((len(subset) / n) * np.log2(len(subset) / n) for subset in partition.values() if len(subset) > 0)
    if intrinsic_info == 0:
        return 0
    return info_gain / intrinsic_info