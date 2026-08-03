def subsets(nums: list): # without duplicates
    result = []
    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)): # start prevents reusing the same element twice
            # make choice
            path.append(nums[i])
            backtrack(i+1, path)
            path.pop()
    backtrack(0, [])
    return result        


if __name__ == "__main__":
    nums = [1,2,3,4]
    print(subsets(nums))