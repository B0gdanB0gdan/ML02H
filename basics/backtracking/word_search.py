"""
Given an m x n grid of characters board and a string word, return true if word exists in the grid.
The word can be constructed from letters of sequentially adjacent cells, where adjacent cells 
are horizontally or vertically neighboring. The same letter cell may not be used more than once.
"""

def word_search(board: list[list[str]], word: str) -> bool:
    rows, cols = len(board), len(board[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    def backtrack(r, c, i):
        if board[r][c] != word[i]:
            return False

        if i == len(word)-1:
            return True

        temp = board[r][c]
        board[r][c] = '#' # mark as visited temporarily

        for dx, dy in directions:
            new_r = r + dx
            new_c = c + dy
            if 0 <= new_r < rows and 0 <= new_c < cols and board[new_r][new_c] != '#':
                if backtrack(new_r, new_c, i+1):
                    board[r][c] = temp
                    return True

        board[r][c] = temp
        return False


    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True
    return False


if __name__ == "__main__":
    board = [
        ["A","B","C","E"],
        ["S","F","C","S"],
        ["A","D","E","E"]
    ]
    word = "ABCCED"
    print(word_search(board, word))
    word = "ABCED"
    print(word_search(board, word))