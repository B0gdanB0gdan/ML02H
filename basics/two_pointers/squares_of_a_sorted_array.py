"""
Given an integer array nums sorted in non-decreasing order, return an array of the squares of each number 
sorted in non-decreasing order.
"""


def sorted_squares(nums: list[int]):
    n = len(nums)
    l, r = 0, n-1
    res = []
    while l <= r:
        sql = nums[l]**2
        sqr = nums[r]**2
        if sql >= sqr:
            res.append(sql)
            l += 1
        else:
            res.append(sqr)
            r -= 1

    res.reverse() # O(n)
    return res


if __name__ == "__main__":
    print(sorted_squares(
        nums=[-4,-1,0,3,10]
    ))