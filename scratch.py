import pickle
import logging
import os
import numpy as np

dataset_path = "./data_zoo/ppo-LunarLander-v2-50000.pkl"
actions = "./data_zoo/LunarLander-v2/ppo-LunarLander-v2-500-2.pkl"

dataset_file = open(dataset_path,'rb')
dataset = pickle.load(dataset_file)
dataset_file.close()

actions_file = open(actions,'rb')
dataset2 = pickle.load(actions_file)
actions_file.close()

for key in list(dataset.keys()):
    d = dataset[key]
    a = dataset2[key]
    
    c = np.concatenate((d,a), axis=0)
    
    dataset[key] = c

filename = "ppo-LunarLander-v2-500-2-combined.pkl"
save_path = "./data_zoo/LunarLander-v2"
logging.info(f"Saving datapoints to {save_path}/{filename}...")
with open(os.path.join(save_path, filename), 'wb') as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)