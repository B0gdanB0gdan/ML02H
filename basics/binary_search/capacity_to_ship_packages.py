"""
A conveyor belt has packages that must be shipped from one port to another within days days.
The ith package on the conveyor belt has a weight of weights[i]. 
Each day, we load the ship with packages on the conveyor belt (in the order given by weights).
We may not load more weight than the maximum weight capacity of the ship.

Return the least weight capacity of the ship that will result in all the packages on the conveyor belt 
being shipped within days days.

Input: weights = [1,2,3,4,5,6,7,8,9,10], days = 5
Output: 15
Explanation: A ship capacity of 15 is the minimum to ship all the packages in 5 days like this:
1st day: 1, 2, 3, 4, 5
2nd day: 6, 7
3rd day: 8
4th day: 9
5th day: 10

Note that the cargo must be shipped in the order given, so using a ship of capacity 14 and 
splitting the packages into parts like (2, 3, 4, 5), (1, 6, 7), (8), (9), (10) is not allowed.

Input: weights = [3,2,2,4,1,4], days = 3
Output: 6
Explanation: A ship capacity of 6 is the minimum to ship all the packages in 3 days like this:
1st day: 3, 2
2nd day: 2, 4
3rd day: 1, 4

Input: weights = [1,2,3,1,1], days = 4
Output: 3
Explanation:
1st day: 1
2nd day: 2
3rd day: 3
4th day: 1, 1
"""

def ship_within_days(weights: list[int], days: int) -> int:
    """
    weights = [1,2,3,1,1], days=2
    max(weights)
    c=3 -> 3 == days
    c=4 -> 3 == days
    c=5 -> 2 == days
    ...
    c=8 -> 1 <= days
    sum(weights)
    """

    def num_days(weights: int, capacity: int) -> int:
        # simulate the loading
        d = 1
        running_capacity = capacity
        i, n = 0, len(weights)
        we = []
        y = []
        while i < n:
            if running_capacity - weights[i] >= 0:
                y.append(weights[i])
                running_capacity -= weights[i]
                i += 1
            else:
                we.append(y)
                y = []
                running_capacity = capacity
                d += 1
        return d


    left, right = max(weights), sum(weights) 
    while left < right:
        mid = (left + right) // 2
        if num_days(weights, mid) <= days:
            # try a smaller or equal capacity
            right = mid
        else:
            # we need more days for sure so try a larger capacity
            left = mid+1

    return left


if __name__ == "__main__":
    print(ship_within_days(weights=[1,2,3,4,5,6,7,8,9,10], days = 5))