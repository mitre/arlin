import gymnasium as gym
import logging

from arlin.dataset import XRLDataset
from arlin.dataset.collectors import RandomDataCollector, BaseDatapoint

from arlin.generation import generate_clusters, generate_embeddings

import arlin.utils.saving_loading as sl_utils

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning) 

env = gym.make("LunarLander-v2", render_mode='rgb_array')
        
collector = RandomDataCollector(datapoint_cls=BaseDatapoint, environment=env)
dataset = XRLDataset(env, collector=collector)

dataset_path = f"/nfs/lslab2/arlin/data_zoo/LunarLander-v2/ppo-50000-env.npz"

# dataset.fill(num_datapoints=50000)
# dataset.save(file_path=dataset_path)
dataset.load(dataset_path)