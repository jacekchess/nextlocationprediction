import random
import pandas as pd
import numpy as np
import os

# Define the categories
categories = ["Home", "Work", "Restaurant", "Gym", "Park", "Shop"]  # Add more categories if needed

# Define the number of users and sequence length
num_paths = 2000
sequence_length = 100

if not os.path.exists('paths/train'):
    os.makedirs('paths/train')
if not os.path.exists('paths/test'):
    os.makedirs('paths/test')
    
# Generate sequences for each user
for i in range(1, num_paths+1):
    user_id = random.choice(range(1,1001))
    sequence = [random.choice(categories) for _ in range(sequence_length)]
    df = pd.DataFrame()
    df.loc[user_id, 'Sequence'] = sequence
    df = df.reset_index(names=['UserID']).explode('Sequence')
    if i <= num_paths*0.7:
        np.save(f"paths/train/path_{i}.npy", df.to_numpy())
    else:
        np.save(f"paths/test/path_{i}.npy", df.to_numpy())