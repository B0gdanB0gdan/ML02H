"""
There is an m x n rectangular island that borders both the Pacific Ocean and Atlantic Ocean. 
The Pacific Ocean touches the island's left and top edges, 
and the Atlantic Ocean touches the island's right and bottom edges.

The island is partitioned into a grid of square cells. 
You are given an m x n integer matrix heights where heights[r][c] represents the height above sea level of the cell at coordinate (r, c).

The island receives a lot of rain, and the rain water can flow to neighboring cells directly north, south, east, and west if the neighboring cell's height is less than or equal to the current cell's height. 
Water can flow from any cell adjacent to an ocean into the ocean.

Return a 2D list of grid coordinates result where result[i] = [ri, ci] denotes that rain water can flow from cell (ri, ci) to both the Pacific and Atlantic oceans.

Input: heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
Output: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
"""

from collections import deque


def pacific_atlantic(heights: list[list[int]]):
    m = len(heights)
    n = len(heights[0])

    dimensions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
    def traverse(start: list) -> set:
        visited = set()
        queue = deque(start)
        order = set()
        while queue:
            i, j = queue.popleft()

            order.add((i, j))
            visited.add((i, j))

            for dx, dy in dimensions:
                new_i, new_j = i + dx, j + dy
                if 0 <= new_i < m and 0 <= new_j < n and (new_i, new_j) not in visited:
                    if heights[new_i][new_j] >= heights[i][j]:
                        queue.append((new_i, new_j)) 
            
        return order
    
    start_pacific = [(0, i) for i in range(n)] + [(j, 0) for j in range(m)]
    start_atlantic = [(m-1, i) for i in range(n)] + [(j, n-1) for j in range(m)]

    visited_pacific = traverse(start_pacific)
    visited_atlantic = traverse(start_atlantic)

    return list(visited_atlantic.intersection(visited_pacific))

if __name__ == "__main__":
    print(pacific_atlantic(
        heights=[
            [1,2,2,3,5],
            [3,2,3,4,4],
            [2,4,5,3,1],
            [6,7,1,4,5],
            [5,1,1,2,4]
        ]
    ))