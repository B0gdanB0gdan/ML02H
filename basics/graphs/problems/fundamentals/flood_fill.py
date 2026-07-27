"""
You are given an image represented by an m x n grid of integers image, 
where image[i][j] represents the pixel value of the image. 
You are also given three integers sr, sc, and color.
 Your task is to perform a flood fill on the image starting from the pixel image[sr][sc].

To perform a flood fill:
Begin with the starting pixel and change its color to color.
Perform the same process for each pixel that is directly adjacent (pixels that share a side with the original pixel, either horizontally or vertically) and shares the same color as the starting pixel.
Keep repeating this process by checking neighboring pixels of the updated pixels and modifying their color if it matches the original color of the starting pixel.

The process stops when there are no more adjacent pixels of the original color to update.
Return the modified image after performing the flood fill.

Input: image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, color = 2

Output: [[2,2,2],[2,2,0],[2,0,1]]
"""


def flood_fill(image: list[list[int]], sr: int, sc: int, color: int) -> list[list[int]]:
    """
    Time: O(m*n)
    Space: O(m*n)
    """
    m = len(image)
    n = len(image[0])
    directions = [(-1,0),(1,0),(0,1),(0,-1)]
    def fill(image, i, j, color, visited):

        init_color = image[i][j]
        image[i][j] = color
        visited.add((i, j))

        for dx, dy in directions:
            new_i, new_j = i+dx, j+dy
            if new_i < m and new_j < n and new_i >= 0 and new_j >= 0 and (new_i, new_j) not in visited:
                if image[new_i][new_j] == init_color:
                    fill(image, new_i, new_j, color, visited)

    fill(image, sr, sc, color, set())
    return image

if __name__ == "__main__":
    image = [[1,1,1],[1,1,0],[1,0,1]]
    sr = 1
    sc = 1
    color = 2
    print(flood_fill(image, sr, sc, color))