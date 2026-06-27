from collections import Counter

from decision_tree.classification_tree.core import DecisionNode, LeafNode, gain_ratio, partition_by


def c45c(samples, attributes, target_attribute, min_samples=5, max_depth=3):
    return _c45c(samples, attributes, target_attribute, depth=0, min_samples=min_samples, max_depth=max_depth)

def _c45c(samples, attributes, target_attribute, depth, min_samples=5, max_depth=3):
    """C4.5 algorithm for classification.
    Extension of ID3 that handles both continuous and discrete attributes,
    deals with missing attribute values, and prunes trees after creation.
    Greedy algorithm: chooses the most immediate best option even if there might be a better tree with a worse-looking
    first move.
    Algorithm:
        *
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

    # split greedy based on the attribute with max gain ratio
    best_attribute = max(attributes, key=lambda attr: gain_ratio(samples=samples,
                                                                 attribute=attr,
                                                                 target_attribute=target_attribute))
    partitions = partition_by(samples, best_attribute)
    new_attributes = [a for a in attributes if a != best_attribute]

    # Recursively build the subtrees
    subtrees = {attribute_value: _c45c(subset,
                                       new_attributes,
                                       target_attribute,
                                       depth=depth + 1,
                                       min_samples=min_samples,
                                       max_depth=max_depth)
                for attribute_value, subset in partitions.items()}
    return DecisionNode(best_attribute, subtrees, default_value=most_common_label)   