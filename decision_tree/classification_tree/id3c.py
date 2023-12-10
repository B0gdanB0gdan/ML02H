"""Decision Tree

Resources:
    Data Science from Scratch: First Principles with Python
Pros:
    Easy to interpret as the process by which they arrive at a prediction is transparent.
    Requires little data preparation.
Cons:
    Not suitable for large datasets. Might easily overfit to training data.
    Not smooth nor continuous -> not good at extrapolation.
Use cases:
    Data is a mix of numeric and categoical attributes.
    Data can be labeled or not.
    Binary or Multiclass classification tasks.

"""
from collections import Counter, defaultdict
from typing import List, Any, NamedTuple, Union, Dict

import numpy as np
import matplotlib.pyplot as plt
import argparse


class LeafNode(NamedTuple):
    value: Any


class DecisionNode(NamedTuple):
    attribute: str
    subtrees: dict
    default_value: Any = None


DecisionTree = Union[LeafNode, DecisionNode]


def add_nums(param1, param2):
    """This method will be used to compare two numbers

    Args:
        param1 (int): The first parameter.
        param2 (int): The second parameter.

    Returns:
        bool: The return value. True if param1 > param2, False otherwise.
    """
    answer = param1 > param2
    return answer


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
    capability. Partitioning by SSN => one person subsets => each person has exactly one label. What happens if we test the model
    on a SSN key different than any value seen during training? The model overfitted to the SSN values seen during training.
    """
    n = sum([len(s) for s in subsets])
    return sum([len(s) / n * data_entropy(s) for s in subsets])


def partition_by(samples, attribute) -> Dict:
    partition = defaultdict(list)
    for sample in samples:
        partition[getattr(sample, attribute)].append(sample)
    return partition


def entropy_of_partition_by(samples, attribute, target_attribute):
    partition = partition_by(samples, attribute)
    labels = [[getattr(sample, target_attribute) for sample in subset]
              for subset in partition.values()]
    return partition_entropy(labels)


def classify(tree, input):
    if isinstance(tree, LeafNode):
        return tree.value

    attr_val = getattr(input, tree.attribute)

    # no subtree for value of the attribute
    if attr_val not in tree.subtrees:
        return tree.default_value

    subtree = tree.subtrees[attr_val]
    return classify(subtree, input)


def id3c(samples, attributes, target_attribute, min_samples=5, max_depth=3):
    return _id3c(samples, attributes, target_attribute, depth=0, min_samples=min_samples, max_depth=max_depth)


def _id3c(samples, attributes, target_attribute, depth, min_samples=5, max_depth=3):
    """Iterative Dichotomiser 3
    Greedy algorithm: chooses the most immediate best option even if there might be a better tree with a worse-looking
    first move.
    Algorithm:
        * if all the samples have the same label -> create leaf node predicting that label
        * if there is no attribute to split on -> create leaf node predicting majority label
        * otherwise, partition by lowest entropy attribute -> create decision node
        * recur on each partitioned subset using remaining attributes
    """
    # Count target labels i.e. how many True and how many False labels
    label_counts = Counter(getattr(sample, target_attribute) for sample in samples)
    # most_common returns a list of tuples
    most_common_label = label_counts.most_common(1)[0][0]

    # If there's a unique label, predict it
    c1 = len(label_counts) == 1

    # If no split attributes left, return the majority label
    c2 = not attributes

    # If remaining samples are less than min_examples
    c3 = len(samples) < min_samples

    # If depth reached max_depth
    c4 = depth == max_depth

    if c1 or c2 or c3 or c4:
        return LeafNode(most_common_label)

    # split greedy based on the attribute with the lowest entropy
    best_attribute = min(attributes, key=lambda attr: entropy_of_partition_by(samples=samples,
                                                                              attribute=attr,
                                                                              target_attribute=target_attribute))
    partitions = partition_by(samples, best_attribute)
    new_attributes = [a for a in attributes if a != best_attribute]

    # Recursively build the subtrees
    subtrees = {attribute_value: _id3c(subset,
                                       new_attributes,
                                       target_attribute,
                                       depth=depth + 1,
                                       min_samples=min_samples,
                                       max_depth=max_depth)
                for attribute_value, subset in partitions.items()}
    return DecisionNode(best_attribute, subtrees, default_value=most_common_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', nargs='?', const=1, type=int)
    args = parser.parse_args()
    # entropy_details()
    print(data_entropy(["a", "a"]))
    print(data_entropy([0, 0, 1, 1]))
    print(data_entropy([0, 0, 1, 1, 2, 2]))
