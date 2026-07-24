"""
A path in a binary tree is a sequence of nodes where each pair of adjacent nodes
in the sequence has an edge connecting them. 
A node can only appear in the sequence at most once. 
Note that the path does not need to pass through the root.

The path sum of a path is the sum of the node's values in the path.

Given the root of a binary tree, return the maximum path sum of any non-empty path.

Input: root = [1,2,3]
Output: 6
Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.
"""

from basics.trees.core import TreeNode
from basics.trees.transform.serialize import deserialize, serialize


def max_path_sum(root: TreeNode):
    """
    Time: O(n)
    Space: O(4h)
    """
    def mps(root, max_val):
        if not root:
            return [0, 0, 0, 0]

        left = mps(root.left, max_val)
        right = mps(root.right, max_val)

        choices = [
            root.val, # short circuit
            root.val + max(left[0], left[1], left[2]),
            root.val + max(right[0], right[1], right[2]),
            root.val + max(left[0], left[1], left[2]) + max(right[0], right[1], right[2])
        ]
        max_val[0] = max(max_val[0], max(choices))
        return choices

    max_val = [float("-inf")]
    mps(root, max_val)
    return max_val[0]


def max_path_sum_v2(root: TreeNode):
    def mps(root, max_val):

        if not root:
            return 0

        # if a child is negative it will only make the path sum worse
        l = max(0, mps(root.left, max_val)) 
        r = max(0, mps(root.right, max_val))

        max_val[0] = max(max_val[0], l + r + root.val)

        return root.val + max(l, r)

    max_val = [float("-inf")]
    mps(root, max_val)
    return max_val[0]



if __name__ == "__main__":
    print(max_path_sum_v2(deserialize([-10,9,20,None,None,15,7])))
    assert max_path_sum(deserialize([-10,9,20,None,None,15,7])) == 42
    assert max_path_sum(deserialize([1,2,3])) == 6
