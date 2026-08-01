"""
Given n non-negative integers representing an elevation map where the width of each bar is 1, 
compute how much water it can trap after raining.

Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.

Input: height = [4,2,0,3,2,5]
Output: 9
"""


def trap(height: list[int]) -> int:
    left, right = 0, len(height)-1
    left_max, right_max = height[left], height[right]
    total_vol = 0
    while left < right:
        if left_max <= right_max:
            left_max = max(left_max, height[left])
            total_vol += max(0, min(left_max, right_max)-height[left])
            left += 1
        else:
            right_max = max(right_max, height[right])
            total_vol += max(0, min(left_max, right_max)-height[right])
            right -= 1
    return total_vol


if __name__ == "__main__":
    print(trap([0,1,0,2,1,0,1,3,2,1,2,1]))
    print(trap([4,2,0,3,2,5]))