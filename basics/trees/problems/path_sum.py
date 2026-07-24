"""
Given the root of a binary tree and an integer targetSum, 
return true if the tree has a root-to-leaf path such that adding up 
all the values along the path equals targetSum.

A leaf is a node with no children.
Input: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
Output: true
Explanation: The root-to-leaf path with the target sum is shown.
"""

from basics.trees.core import TreeNode
from basics.trees.transform.serialize import deserialize, serialize


def path_sum(root: TreeNode, target_sum: int):
    """
    Time: O(n)
    Space: O(h)
    """
    if not root:
        return False
    if root.left is None and root.right is None:
        return target_sum == root.val
    
    left = path_sum(root.left, target_sum-root.val)
    right = path_sum(root.right, target_sum-root.val)

    return left or right


if __name__ == "__main__":
    target_sum = 22
    print(path_sum(deserialize([5,4,8,11,None,13,4,7,2,None,None,None,1]), target_sum))
    target_sum = 5
    print(path_sum(deserialize([1,2,3]), target_sum))
    print(path_sum(TreeNode(1, left=TreeNode(2)), 1))