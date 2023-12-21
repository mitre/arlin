import os
from typing import List, Union

from PIL import Image


def get_model_name(model_type: str, freq: int, threshold: int) -> str:
    model_name = model_type.lower().capitalize()

    if model_name in ["Random", "Adversarial"]:
        return f"{model_name}_{freq}"
    elif model_name == "Preference":
        return f"{model_name}_{threshold}"
    else:
        raise ValueError(f"Model type {model_type} not supported.")


def create_dirs(base_dir_path: str, model_names: List[str] = []) -> None:
    gifs_dir_name = os.path.join(base_dir_path, "gifs")
    metrics_dir_name = os.path.join(base_dir_path, "metrics")
    cosine_dir_name = os.path.join(metrics_dir_name, "cosine_similarity")
    rewards_dir_name = os.path.join(metrics_dir_name, "episode_rewards")

    os.makedirs(gifs_dir_name, exist_ok=True)
    os.makedirs(metrics_dir_name, exist_ok=True)
    os.makedirs(cosine_dir_name, exist_ok=True)
    os.makedirs(rewards_dir_name, exist_ok=True)

    for name in model_names:
        os.makedirs(os.path.join(gifs_dir_name, name), exist_ok=True)


def save_gifs(
    gif_lists: List[List[Image.Image]],
    episode_rewards: List[Union[int, float]],
    dir_name: str,
):
    idx = episode_rewards.index(max(episode_rewards))
    save_path = os.path.join(dir_name, f"episode_{idx}-max.gif")
    gif_lists[idx][0].save(
        save_path, save_all=True, append_images=gif_lists[idx], duration=30, loop=0
    )

    idx = episode_rewards.index(min(episode_rewards))

    save_path = os.path.join(dir_name, f"episode_{idx}-min.gif")
    gif_lists[idx][0].save(
        save_path, save_all=True, append_images=gif_lists[idx], duration=30, loop=0
    )
