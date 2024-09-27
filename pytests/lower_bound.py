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

import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [10, 15, 20, 25, 30]  # Same data for both axes

# Create the first plot
fig, ax1 = plt.subplots()

# Plot the data on the first y-axis
ax1.set_xlabel('X Axis')
ax1.set_ylabel('Y1 Axis', color='blue')
ax1.plot(x, y, color='blue', label='Y1 Data')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis with a different scale
ax2 = ax1.twinx()
ax2.set_ylabel('Y2 Axis (different scale)', color='green')

# Optionally rescale y for ax2, e.g., by multiplying by a constant
y_rescaled = [value * 2 for value in y]  # Example of rescaling

# Plot the same data (but scaled) on the second y-axis
ax2.plot(x, y_rescaled, color='green', label='Y2 Data (rescaled)')
ax2.tick_params(axis='y', labelcolor='green')

# Show plot
plt.title('Plot with Two Y-Axes (Different Scales)')
plt.show()