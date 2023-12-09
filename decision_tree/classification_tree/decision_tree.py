"""Decision Tree

Resources:
    Data Science from Scratch: First Principles with Python
Pros:
    Easy to interpret as the process by which they arrive at a prediction is transparent.
Cons:
    Not suitable for large datasets. Might easily overfit to training data.
Use case:
    Data is a mix of numeric and categoical attributes. Data can be labeled or not.
"""
from collections import Counter, defaultdict
from typing import List, Any

import numpy as np
import matplotlib.pyplot as plt
import argparse


def add_nums(num1, num2):
    """This method will be used to add two numbers

        :param int num1: The first number
        :param int num2: The second number

        :returns: The sum of two numbers

        :rtype: int
    """
    answer = num1 + num2
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
    return [c/len(labels) for c in count]


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
    return [len(s) / n * data_entropy(s) for s in subsets]


def partition_by(samples, attribute):
    partition = defaultdict(list)
    for sample in samples:
        partition[getattr(sample, attribute)].append(sample)
    return partition


def entropy_of_partition_by(samples, attribute, target_attribute):
    partition = partition_by(samples, attribute)
    labels = [[getattr(sample, target_attribute) for sample in subset]
              for subset in partition]
    return partition_entropy(labels)


def id3():
    """
    Greedy algorithm: chooses the most immediate best option even if there wight be a better tree with a worse-looking
    first move.
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', nargs='?', const=1, type=int)
    args = parser.parse_args()
    # entropy_details()
    print(data_entropy(["a", "a"]))
    print(data_entropy([0, 0, 1, 1]))
    print(data_entropy([0, 0, 1, 1, 2, 2]))



