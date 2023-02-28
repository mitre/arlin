from sklearn.manifold import TSNE
import argparse
import pickle

from perfect_timing.perfect_timing.utils.config_utils import get_config

def main(config):
    with open(config['pkl_path'], 'rb') as f:
        data = pickle.load(f)
    
    print(data.keys())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        'config_path', 
        type=str,
        help='Path to config.'
        )
    
    args = parser.parse_args()
    config = get_config(args.config_path)
    main(config)