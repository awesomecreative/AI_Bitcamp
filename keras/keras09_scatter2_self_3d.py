import matplotlib.pyplot as plt

# Generate some 3D data
x = [1, 2, 3, 4, 5]
y = [2, 3, 4, 5, 6]
z = [1, 4, 9, 16, 25]

# Create a figure and a 3D scatter plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c=z, cmap='Reds')

plt.show()
