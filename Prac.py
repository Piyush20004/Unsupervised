import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# 1. Load Vertebral Column Dataset
# --------------------------------------------------
# Make sure "column_3C.dat" is saved exactly in the same folder as your Python script
try:
    data = pd.read_csv("column_3C.dat", sep=r"\s+", header=None)
except FileNotFoundError:
    print("ERROR: Could not find 'column_3C.dat'. Please ensure it is saved on your Desktop next to Prac.py")
    exit()

print(f"Dataset Shape: {data.shape}\n")
print("First 5 Records:")
print(data.head())

# --------------------------------------------------
# 2. Separate Features and Class
# --------------------------------------------------
X = data.iloc[:, 0:6].values
y = data.iloc[:, 6].astype(str).str.strip().values 

# --------------------------------------------------
# 3. Normalize Data
# --------------------------------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("\nNormalized Data:")
print(X_scaled[:5])

# --------------------------------------------------
# 4. Create SOM
# --------------------------------------------------
som = MiniSom(
    x=3,
    y=2,
    input_len=6,
    sigma=1.0,
    learning_rate=0.5,
    random_seed=42
)

# --------------------------------------------------
# 5. Initialize & Train SOM
# --------------------------------------------------
som.random_weights_init(X_scaled)
som.train_random(X_scaled, 1000)

print("\nSOM Training Completed.")

# --------------------------------------------------
# 6. Quantization Error
# --------------------------------------------------
error = som.quantization_error(X_scaled)
print(f"Quantization Error: {error}")

# --------------------------------------------------
# 7. Display Classes on SOM U-Matrix
# --------------------------------------------------
plt.figure(figsize=(8, 6))

plt.pcolor(som.distance_map().T)
plt.colorbar(label="Distance")

markers = {"DH": "o", "SL": "s", "NO": "^"}
colors = {"DH": "red", "SL": "blue", "NO": "green"}

for i, sample in enumerate(X_scaled):
    winner = som.winner(sample)
    
    plt.plot(
        winner[0] + 0.5,
        winner[1] + 0.5,
        markers[y[i]],
        markerfacecolor="None",
        markeredgecolor=colors[y[i]],
        markersize=10,
        markeredgewidth=2
    )

plt.title("SOM U-Matrix")
plt.xlabel("SOM X")
plt.ylabel("SOM Y")

plt.xlim(0, 3)
plt.ylim(0, 2)

plt.show()
