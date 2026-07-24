"""
Given the roots of two binary trees p and q,
write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, 
and the nodes have the same value.
"""

from basics.trees.core import TreeNode
from basics.trees.transform.serialize import deserialize, serialize


def same_tree(p: TreeNode, q: TreeNode):

    if not p and not q:
        return True

    if not p or not q:
        return False

    l = same_tree(p.left, q.left)
    r = same_tree(p.right, q.right)

    return l and r and p.val == q.val


if __name__ == "__main__":
    print(same_tree(
        deserialize([1, 2]),
        deserialize([1, None, 2])
    ))