# %%
import json
from transform.mongo.mongo_pepwave_transform import MongoPepwaveTransform
from pathlib import Path

from datetime import datetime

print("-" * 200)
print("Current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("-" * 200)

with open(Path(__file__).parent / "intermediate_output.json") as f:
    intermediate_output = json.load(f)

tf = MongoPepwaveTransform()
res = tf.filter_transformed_docs(intermediate_output["docs"])

print(len(res))

# %%
import numpy as np
from sklearn.preprocessing import QuantileTransformer

num_zeros = 10000
num_random_numbers = 8000
# Create an array of random numbers between 1 and 20
random_numbers = np.random.randint(1, 21, size=num_random_numbers)

# Convert random_numbers to a list before concatenation
arr = (
    [1, 2, 9, 10, 11, 18, 19, 20]
    + random_numbers.tolist()
    + ([0] * num_zeros)
    # + [20] * 5000
)
feature_matrix = np.array([arr]).T  # Transpose to make shape (n, 1)

scaler = QuantileTransformer(
    n_quantiles=1000,  # Adjust based on dataset size
    output_distribution="uniform",
)
scaled_features = scaler.fit_transform(feature_matrix)
print(scaled_features[0:8, 0])
# Calculate the mean of the scaled features column
mean_scaled_features = np.mean(scaled_features[:, 0])
print(f"Mean of scaled features: {mean_scaled_features}")
