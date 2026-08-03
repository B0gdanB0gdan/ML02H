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