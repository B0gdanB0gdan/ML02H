"""
Given the root of a binary tree, return the length of the diameter of the tree.

The diameter of a binary tree is the length of the longest path between any two nodes in a tree. 
This path may or may not pass through the root.

The length of a path between two nodes is represented by the number of edges between them.
"""

from basics.trees.core import TreeNode
from basics.trees.transform.serialize import deserialize, serialize


def diameter(root: TreeNode):
    """
    Time: O(n)
    Space: O(h)
    """
    max_diam = [0]
    def diam(root):
        if not root:
            return 0
        
        height_left = diam(root.left)
        height_right = diam(root.right)

        max_diam[0] = max(max_diam[0], height_left + height_right)

        return 1 + max(height_left, height_right)

    diam(root)
    return max_diam[0]
    

if __name__ == "__main__":
    arr = root = [1,2,3,4,5]
    root = deserialize(arr)
    print(diameter(root))