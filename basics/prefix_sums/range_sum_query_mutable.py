"""
Given an integer array nums, handle multiple queries of the following types:
Update the value of an element in nums.
Calculate the sum of the elements of nums between indices left and right inclusive where left <= right.
Implement the NumArray class:

NumArray(int[] nums) Initializes the object with the integer array nums.
void update(int index, int val) Updates the value of nums[index] to be val.
int sumRange(int left, int right) Returns the sum of the elements of nums between indices left and right inclusive (i.e. nums[left] + nums[left + 1] + ... + nums[right]).

Input
["NumArray", "sumRange", "update", "sumRange"]
[[[1, 3, 5]], [0, 2], [1, 2], [0, 2]]
Output
[null, 9, null, 8]

Explanation
NumArray numArray = new NumArray([1, 3, 5]);
numArray.sumRange(0, 2); // return 1 + 3 + 5 = 9
numArray.update(1, 2);   // nums = [1, 2, 5]
numArray.sumRange(0, 2); // return 1 + 2 + 5 = 8
"""
from .fenwick_tree import FenwickTree


class NumArray:
    def __init__(self, nums: list[int]):
        self.n = len(nums)
        self.tree = FenwickTree(self.n)
        self.nums = nums
        for i, num in enumerate(nums):
            self.tree.update(i+1, num)

    def update(self, index: int, val: int) -> None:
        delta = val - self.nums[index]
        self.tree.update(index, delta)

    def sum_range(self, left: int, right: int) -> int:
        return self.tree.range_query(left+1, right+1)


if __name__ == "__main__":
    obj = NumArray([1, 3, 5])
    print(obj.sum_range(0, 2))
    obj.update(1, 2)
    print(obj.sum_range(0, 2))