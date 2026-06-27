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


import argparse
from collections import Counter
from decision_tree.classification_tree.core import DecisionNode, LeafNode, data_entropy, entropy_details, information_gain, partition_by


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
        * otherwise, partition by lowest entropy attribute or max information gain -> create decision node
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

    # split greedy based on the attribute with max information gain
    best_attribute = max(attributes, key=lambda attr: information_gain(samples=samples,
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
    entropy_details()
    print(data_entropy(["a", "a"]))
    print(data_entropy([0, 0, 1, 1]))
    print(data_entropy([0, 0, 1, 1, 2, 2]))
