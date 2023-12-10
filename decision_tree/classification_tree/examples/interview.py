from typing import NamedTuple, Optional

from decision_tree.classification_tree.id3c import id3c, classify
from decision_tree.dt_utils.plot import plot_tree


class Candidate(NamedTuple):
    level: str
    lang: str
    tweets: bool
    phd: bool
    did_well: Optional[bool] = None  # allow unlabeled data


# level lang tweets phd did_well
interview_data = [
    Candidate('Senior', 'Java', False, False, False),
    Candidate('Senior', 'Java', False, True, False),
    Candidate('Mid', 'Python', False, False, True),
    Candidate('Junior', 'Python', False, False, True),
    Candidate('Junior', 'R', True, False, True),
    Candidate('Junior', 'R', True, True, False),
    Candidate('Mid', 'R', True, True, True),
    Candidate('Senior', 'Python', False, False, False),
    Candidate('Senior', 'R', True, False, True),
    Candidate('Junior', 'Python', True, False, True),
    Candidate('Senior', 'Python', True, True, True),
    Candidate('Mid', 'Python', False, True, True),
    Candidate('Mid', 'Java', True, False, True),
    Candidate('Junior', 'Python', False, True, False)
]


if __name__ == "__main__":
    tree = id3c(interview_data, ['level', 'lang', 'tweets', 'phd'], 'did_well')
    print(tree)
    print(classify(tree, Candidate("Junior", "Java", True, False)))
    print(classify(tree, Candidate("Junior", "Java", True, True)))
    print(classify(tree, Candidate("Intern", "Java", True, True)))
    plot_tree(tree)


