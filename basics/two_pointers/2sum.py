"""
Variation of leet code problem but assumes array is sorted
"""

def two_sum(nums, target):
    """
        Oposite direction pointers
        Use when:
        - array is already sorted
        - looking for a pair/triplet
        - to satisfy a (e.g., sum, comparison based) condition
    """

    left, right = 0, len(nums)-1
    while left < right:
        current = nums[left] + nums[right]
        if current == target:
            return [left, right]
        elif current < target:
            left += 1
        else:
            right -= 1
    return [-1, -1]


if __name__ == "__main__":
    print(two_sum(
        nums=[2,7,11,15], target = 9
    ))