"""
Koko loves to eat bananas. 
There are n piles of bananas, the ith pile has piles[i] bananas. 
The guards have gone and will come back in h hours.
Koko can decide her bananas-per-hour eating speed of k. 
Each hour, she chooses some pile of bananas and eats k bananas from that pile. 
If the pile has less than k bananas, she eats all of them instead and will not eat any more bananas during this hour.
Koko likes to eat slowly but still wants to finish eating all the bananas before the guards return.
Return the minimum integer k such that she can eat all the bananas within h hours.

Input: piles = [3,6,7,11], h = 8
Output: 4
Example 2:

Input: piles = [30,11,23,4,20], h = 5
Output: 30
Example 3:

Input: piles = [30,11,23,4,20], h = 6
Output: 23
"""

def min_eating_speed(piles: list[int], h: int) -> int:
    def hours_needed(speed):
        return sum((pile + speed - 1) // speed for pile in piles)  # ceil division

    left, right = 1, max(piles)
    while left < right: # the moment left = right is the moment of switch
        mid = (left + right) // 2
        if hours_needed(mid) <= h:
            # ate too fast, slow down
            right = mid
        else:
            # ate too slow, increase speed
            left = mid+1
    return left

if __name__ == "__main__":
    print(min_eating_speed(piles=[3,6,7,11], h=8))