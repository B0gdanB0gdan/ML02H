from typing import NamedTuple, Optional

from decision_tree.dt_utils.plot import plot_tree
from decision_tree.regression_tree.id3r import id3r, predict


class Candidate(NamedTuple):
    level: int  # 0=Senior, 1=Mid, 2=Junior
    lang: int  # 0=Java, 1=Python, 2=R
    tweets: int  # 0=False, 1=True
    phd: int  # 0=False, 1=True
    age: int
    grade: Optional[float] = None  # allow unlabeled data


# level lang tweets phd did_well
interview_data = [
    Candidate(0, 0, 0, 0, 40, 9),
    Candidate(0, 0, 0, 1, 38, 10),
    Candidate(1, 1, 0, 0, 28, 8.1),
    Candidate(2, 1, 0, 0, 24, 5.5),
    Candidate(2, 2, 1, 0, 25, 6),
    Candidate(2, 2, 1, 1, 22, 4),
    Candidate(1, 2, 1, 1, 27, 7.6),
    Candidate(0, 1, 0, 0, 30, 8.4),
    Candidate(0, 2, 1, 0, 32, 9.5),
    Candidate(0, 1, 1, 0, 31, 5),
    Candidate(0, 1, 1, 1, 34, 10),
    Candidate(1, 1, 0, 1, 27, 7),
    Candidate(1, 0, 1, 0, 27, 8.8),
    Candidate(2, 1, 0, 1, 20, 6)
]

if __name__ == "__main__":
    tree = id3r(interview_data, ['level', 'lang', 'tweets', 'phd', 'age'], 'grade', min_samples=0, max_depth=10)
    print(predict(tree, Candidate(2, 0, 1, 0, 22)))
    print(predict(tree, Candidate(2, 0, 1, 1, 23)))
    print(predict(tree, Candidate(0, 0, 1, 1, 28)))
    plot_tree(tree)
