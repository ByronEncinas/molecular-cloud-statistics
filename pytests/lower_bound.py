import numpy as np

# Function to keep interior False values only
def cut_boundary_falses(row):
    # Find indices of True values
    true_indices = np.where(row == True)[0]
    
    # Check for no True or only one True value
    if len(true_indices) <= 1:
        return row  # Return the row as is for no True values or a single True

    # Get the first and last occurrence of True
    first_true = true_indices[0]
    last_true = true_indices[-1]

    # Create a mask that keeps only the interior True values
    mask = np.zeros_like(row, dtype=bool)  # Start with all False
    
    # Set the values between first and last True to True
    mask[first_true + 1:last_true] = True  # Only preserve interior True values

    return mask

# True means that I want to keep it.

a = np.array([[False,False,True,True,False,True,False,True,True,True,True,True,True,True,False,False,True,False],
              [False,False,True,True,False,True,False,True,True,True,True,True,True,True,False,False,True,False] ])

print(a)
print()

cut_lower_bound = np.array([cut_boundary_falses(row) for row in a])
#cut_lower_bound = np.array([row for row in lower_bound])
print(cut_lower_bound)

