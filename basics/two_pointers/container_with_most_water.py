"""
You are given an integer array height of length n. 
There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).
Find two lines that together with the x-axis form a container, such that the container contains the most water.
Return the maximum amount of water a container can store.
Notice that you may not slant the container.

Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
"""


def max_area(height: list[int]) -> int:

    n = len(height)
    left, right = 0, n-1
    area = 0
    while left < right:
        area = max(area, (right-left) * min(height[left], height[right]))
        if height[left] < height[right]:
            left += 1
        else:
            right -=1
    return area


if __name__ == "__main__":
    print(max_area([1,8,6,2,5,4,8,3,7]))