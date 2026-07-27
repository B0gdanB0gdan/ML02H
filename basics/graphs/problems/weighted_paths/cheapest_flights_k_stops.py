"""
There are n cities connected by some number of flights. 
You are given an array flights where flights[i] = [from_i, to_i, price_i]
indicates that there is a flight from city fromi to city toi with cost pricei.

You are also given three integers src, dst, and k, return the cheapest price from src to dst 
with at most k stops. 
If there is no such route, return -1.

Input: n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]]
src = 0, dst = 3, k = 1
Output: 700
Explanation:
The graph is shown above.
The optimal path with at most 1 stop from city 0 to 3 is marked in red and has cost 100 + 600 = 700.
Note that the path through cities [0,1,2,3] is cheaper but is invalid because it uses 2 stops.

"""


def find_cheapest_price(n: int, flights: list[list[int]], src: int, dst: int, k: int) -> int:

    cost = {src: 0} # dist from src to every key in dict

    for _ in range(k+1):
        new_cost = cost.copy()
        for s, d, w in flights:
            if cost.get(s, float("inf")) + w < cost.get(d, float("inf")):
                new_cost[d] = cost[s] + w
        cost = new_cost
    return -1 if cost[dst] == float("inf") else cost[dst]
    

if __name__ == "__main__":
    flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]]
    print(find_cheapest_price(4, flights, 0, 3, 1))