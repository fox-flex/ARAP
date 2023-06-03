import matplotlib.pyplot as plt

# Initialize lists to store data
iterations = []
energies = []

# Read the file
with open('data/tmp.txt', 'r') as file:
    # Iterate over each line
    for line in file:
        # Check if the line contains iteration and energy information
        if '[Open3D DEBUG] [DeformAsRigidAsPossible] iter=' in line and ', energy=' in line:
            # Extract the iteration number and energy value from the line
            iter_num = int(line.split('iter=')[1].split(',')[0])
            energy = float(line.split('energy=')[1].strip())
            
            # Append the values to the respective lists
            iterations.append(iter_num)
            energies.append(energy)

# Plot the data
plt.plot(iterations, energies)
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.title('Energy vs Iteration')
plt.grid(True)

# Save the plot to a file
plt.savefig('vis/armadillo_energy.png')
