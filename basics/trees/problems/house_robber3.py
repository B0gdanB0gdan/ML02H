"""
The thief has found himself a new place for his thievery again. 
There is only one entrance to this area, called root.

Besides the root, each house has one and only one parent house. 
After a tour, the smart thief realized that all houses in this place form a binary tree.
It will automatically contact the police if two directly-linked houses were broken into on the same night.

Given the root of the binary tree, return the maximum amount of money the thief can rob without alerting the police.

Input: root = [3,2,3,null,3,null,1]
Output: 7
Explanation: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.

"""

from basics.trees.core import TreeNode
from basics.trees.transform.serialize import deserialize, serialize


def house_robber_3(root: TreeNode):
   """
   Time: O(n)
   Space: O(n)
   """
   def rob_rec(root):

      if not root:
         # rob, not rob
         return [0, 0]

      left = rob_rec(root.left)
      right = rob_rec(root.right)

      rob = root.val + left[1] + right[1]
      not_rob = max(left[0], left[1]) + max(right[0], right[1])

      return [rob, not_rob]
   
   return max(rob_rec(root))



if __name__ == "__main__":
    print(house_robber_3(deserialize([3,2,3,None,3,None,1]))) # 7
    print(house_robber_3(deserialize([3,4,5,1,3,None,1]))) # 9

