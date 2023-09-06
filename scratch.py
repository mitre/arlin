import gymnasium as gym

from arlin.dataset import XRLDataset
from arlin.dataset.collectors import RandomDataCollector, BaseDatapoint

from arlin.generation import generate_clusters, generate_embeddings

import arlin.utils.saving_loading as sl_utils

env = gym.make("LunarLander-v2", render_mode='rgb_array')
        
collector = RandomDataCollector(datapoint_cls=BaseDatapoint, environment=env)
dataset = XRLDataset(env, collector=collector)

dataset_path = f"/nfs/lslab2/arlin/data_zoo/LunarLander-v2/ppo-50000-env.npz"

dataset.fill(num_datapoints=50000)
dataset.save(file_path=dataset_path)

embeddings = generate_embeddings(dataset=dataset,
                                activation_key='latent_actors',
                                perplexity=500,
                                n_train_iter=2000,
                                output_dim=2,
                                seed=12345)

sl_utils.save_data(embeddings, f"/nfs/lslab2/arlin/data_zoo/LunarLander-v2/ppo-50000-env-embeddings.npy")