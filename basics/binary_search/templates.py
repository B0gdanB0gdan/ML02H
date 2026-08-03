def binary_search(arr, target):
    """Template 1. Find exact value
    Index where val==target
    """
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


# first and last occurance
def find_first(arr, target):
        left, right = 0, len(arr) - 1
        result = -1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                result = mid
                right = mid - 1        # keep searching LEFT for an earlier match
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return result

def find_last(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            result = mid
            left = mid + 1         # keep searching RIGHT for a later match
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result

# Search in rotated sorted array
# Use when: array was sorted, then rotated at some unknown pivot.
# always do the search (setting left and right) in the sorted part

def search_rotated(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid

        if arr[left] <= arr[mid]:          # LEFT half is sorted
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:                                # RIGHT half is sorted
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1

# Template 4. Binary search on the ANSWER 


if __name__ == "__main__":
    print(binary_search([1,2,3,4,5,6], target=7))