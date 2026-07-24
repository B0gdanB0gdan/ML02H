"""
Given the root of a binary tree, check whether it is a mirror of itself 
(i.e., symmetric around its center).
"""

from basics.trees.core import TreeNode
from basics.trees.transform.serialize import deserialize, serialize


def is_sym(root):

    def check(p, q):
        if not p and not q:
            return True
        if not p or not q:
            return False

        return check(p.left, q.right) and check(p.right, q.left) and p.val == q.val

    return check(root.left, root.right)


if __name__ == "__main__":
    print(is_sym(deserialize([1,2,2,3,4,4,3])))
    print(is_sym(deserialize([1,2,2,None,3,None,3])))