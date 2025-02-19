# Read and plot the contour points and normals
# csherwood@usgs.gov

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dsc = pd.read_csv('contour15.csv')

plt.figure(figsize=(8, 8))

# Plot the smoothed 15-m contour
plt.plot(dsc['contour15s_lon'], dsc['contour15s_lat'], 'k-', label='15-m contour')  # Longitude vs Latitude

# Plot the normals, plot every 20th
plt.quiver(
    dsc['contour15s_lon'][::20], dsc['contour15s_lat'][::20], dsc['normals_x'][::20], dsc['normals_y'][::20],
    color='b', angles='xy', scale_units='xy', scale=2, label='Normals')
plt.legend()
plt.savefig('contour15.png')