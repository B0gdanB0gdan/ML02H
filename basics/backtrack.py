"""

def backtrack(state):
    if solution_found:
        save_answer()
        return

    for choice in choices:
        make_choice(choice)      # DO
        backtrack(new_state)     # EXPLORE
        undo_choice(choice) 

"""

def subsets(nums: list):
    result = []
    def backtrack(subset, i):
        
        if i == len(nums):
            result.append(subset[:])
            return
        
        subset.append(nums[i])
        backtrack(subset, i+1)
        subset.pop()
        backtrack(subset, i+1)
    backtrack([], 0)
    return result        


if __name__ == "__main__":
    nums = [1,2,3,4]
    print(subsets(nums))