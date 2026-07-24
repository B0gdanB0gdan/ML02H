"""
Given a binary tree root, a node X in the tree is named good 
if in the path from root to X there are no nodes with a value greater than X.

Return the number of good nodes in the binary tree.

Input: root = [3,1,4,3,null,1,5]
Output: 4
Explanation: Nodes in blue are good.
Root Node (3) is always a good node.
Node 4 -> (3,4) is the maximum value in the path starting from the root.
Node 5 -> (3,4,5) is the maximum value in the path
Node 3 -> (3,1,3) is the maximum value in the path.
"""

from basics.trees.core import TreeNode
from basics.trees.transform.serialize import deserialize, serialize


def count_good_nodes(root: TreeNode):
    """
    Time: O(n)
    Space: O(h)
    """
    def count_nodes(root: TreeNode, max_val: int, c:list):

        if not root:
            return

        if root.val >= max_val:
            max_val = root.val
            c[0] += 1

        count_nodes(root.left, max_val, c)
        count_nodes(root.right, max_val, c)

    c=[0]
    count_nodes(root, max_val=root.val, c=c)
    return c[0]



if __name__ == "__main__":
    print(count_good_nodes(deserialize([3,1,4,3,None,1,5])))