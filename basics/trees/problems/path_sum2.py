"""
Given the root of a binary tree and an integer targetSum, 
return all root-to-leaf paths where the sum of the node values in the path equals targetSum. 
Each path should be returned as a list of the node values, not node references.

A root-to-leaf path is a path starting from the root and ending at any leaf node. 
A leaf is a node with no children.

Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
Output: [[5,4,11,2],[5,8,4,5]]
Explanation: There are two paths whose sum equals targetSum:
5 + 4 + 11 + 2 = 22
5 + 8 + 4 + 5 = 22
"""


from basics.trees.core import TreeNode
from basics.trees.transform.serialize import deserialize, serialize


def path_sum2(root: TreeNode, target_sum: int):
    """
    Time: O()
    Space: O()
    """
    result = []
    stack = []
    def path_sum(root, target_sum):
        if not root:
            return
        
        stack.append(root.val)

        if root.left is None and root.right is None:
            if root.val == target_sum:
                result.append(stack[:])
        
        path_sum(root.left, target_sum - root.val)
        path_sum(root.right, target_sum - root.val)

        stack.pop()

    path_sum(root, target_sum)
    return result


if __name__ == "__main__":
    print(path_sum2(deserialize([5,4,8,11,None,13,4,7,2,None,None,5,1]), 22))
    print(path_sum2(deserialize([1,2,3]), 5))
