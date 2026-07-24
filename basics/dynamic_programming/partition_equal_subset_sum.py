"""
Given an integer array nums, return true if you can partition the array into two subsets 
such that the sum of the elements in both subsets is equal or false otherwise.
Example 1:

Input: nums = [1,5,11,5]
Output: true
Explanation: The array can be partitioned as [1, 5, 5] and [11].
Example 2:

Input: nums = [1,2,3,5]
Output: false
Explanation: The array cannot be partitioned into equal sum subsets.
"""


def partition_memo_helper(nums, n, s, dp):
    """
    dp[(n, s)] = whether a given sum s can be achieved using the first n elements either by including or excluding an element
    Time: O(n*s)
    Space: O(n*s)
    """
    if s == 0:
        return True
    
    if n == 0:
        return False
    
    if (n, s) in dp:
        return dp[(n, s)]


    dp[(n, s)] =  partition_memo_helper(nums, n-1, s - nums[n-1], dp) or \
                  partition_memo_helper(nums, n-1, s, dp)
    
    return dp[(n, s)]


def partition_memo(nums):
    s = sum(nums)
    if s % 2 != 0:
        return False 
    n = len(nums)
    return partition_memo_helper(nums, n, s//2, {})


def partition_tab(nums):
    """
    dp[(i, j)] = whether a given sum j can be achieved using the first i elements either by including or excluding an element
                 or -> can you achieve half of the sum (j) using first i elements either by incl/excl
    Time: O(n*s)
    Space: O(n*s)
    """
    n = len(nums)
    s = sum(nums) // 2

    if sum(nums) % 2 != 0:
        return False
    
    dp = [[False] * (s+1) for _ in range(n+1)]

    for i in range(n+1):
        dp[i][0] = True

    for i in range(1, n+1):
        for sp in range(1, s+1):
            if sp - nums[i-1] >= 0: 
                dp[i][sp] = ( 
                    dp[i-1][sp-nums[i-1]] # the remaning elements need to cover the rest of the sum
                    or 
                    dp[i-1][sp] # the remaning elements need to cover the whole sum
                )
            else:
                # the element cannot be considered
                # using the elements before this one, was this sum already reachable?
                dp[i][sp] = dp[i-1][sp]

    return dp[n][s]



if __name__ == "__main__":
    nums = [1,5,11,5]
    print("Memo:", partition_memo(nums))
    print("Tab:", partition_tab(nums))
    nums = [1,2,3,5]
    print("Memo:", partition_memo(nums))
    print("Tab:", partition_tab(nums))