import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hockey_rink import NHLRink

# Load data
df = pd.read_csv("C:/Users/neelc/Desktop/shots_ropes_cleaned.csv")

# Extract coordinates
x = df['xCordAdjusted']
y = df['yCordAdjusted']

# Create a DataFrame for plotting
shots = pd.DataFrame({
    'x': x,
    'y': y
})

# Initialize the rink
rink = NHLRink()

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))

# Plot using rink.scatter
rink.scatter("x", "y", facecolor="blue", s=100, edgecolor="white", data=shots, ax=ax)
ax.set_title('Shot Locations for Ropes, 42 shots, 31 SOG, 4.12 5v5 expected goals, 4 5v5 actual goals')

# Show plot
plt.show()
