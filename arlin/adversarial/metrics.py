import logging
import statistics
from copy import deepcopy
from math import floor
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy.special import kl_div
from scipy.stats import norm as stats_norm

from arlin.analysis.visualization import COLORS


def _calculate_cosine_sim(
    base_obs: List[np.ndarray], target_obs: List[np.ndarray]
) -> Tuple[List[int], List[float]]:
    def cosine_sim(a: np.ndarray, b: np.ndarray):
        return round(dot(a, b) / (norm(a) * norm(b)), 6)

    if len(base_obs) < len(target_obs):
        num_extra = len(target_obs) - len(base_obs)
        last_obs_extra = [base_obs[-1]] * num_extra
        baseline_obs_extend = base_obs + last_obs_extra
    else:
        baseline_obs_extend = base_obs

    target_x = list(range(len(target_obs)))
    target_y = [cosine_sim(a, b) for (a, b) in zip(target_obs, baseline_obs_extend)]

    return target_x, target_y


def plot_cosine_sim(
    baseline_obs: List[np.ndarray],
    target_obs: List[List[np.ndarray]],
    names: List[str],
    save_path: str,
) -> None:
    for i in range(len(names)):
        xs, ys = _calculate_cosine_sim(baseline_obs, target_obs[i])
        plt.plot(xs, ys, color=COLORS[i], label=names[i])

    plt.legend()

    plt.xlabel("Timestep")
    plt.ylabel("Cosine Similarity")

    plt.title("Observation Cosine Similarity Over an Episode", fontweight="bold")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def kl_divergence(gt_dist: np.ndarray, adv_action: int) -> float:
    gt_action = np.argmax(gt_dist)
    adv_dist = deepcopy(gt_dist)

    adv_prob = gt_dist[adv_action]
    gt_prob = gt_dist[gt_action]

    adv_dist[gt_action] = adv_prob
    adv_dist[adv_action] = gt_prob

    return sum(kl_div(gt_dist, adv_dist)) / len(gt_dist)


def _mu_sig(divergences: List[float]) -> Tuple[float, float]:
    mu = statistics.mean(divergences)
    sigma = statistics.stdev(divergences)

    return (mu, sigma)


def _max_x(stats: List[Tuple[float, float]]) -> float:
    max_x = 0
    for mu, sigma in stats:
        two_std = mu + (2 * sigma)
        if two_std > max_x:
            max_x = two_std

    return max_x


def plot_divergences(
    divergences: List[List[float]], names: List[str], save_path: str
) -> None:
    stats = [_mu_sig(i) for i in divergences]
    max_x = _max_x(stats)

    x = np.arange(0, max_x, 0.01)

    for i in range(len(names)):
        mu, sigma = stats[i]
        pdf = stats_norm.pdf(x, mu, sigma)

        # Skip 0 because baseline is not included
        color = COLORS[i + 1]
        plt.plot(x, pdf, "--", color=color, label=names[i])

    plt.xlabel("KL Divergence")
    plt.ylabel("Likelihood of Observance")
    plt.legend()

    plt.title("Distribution of Attacked KL Divergences", fontweight="bold")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_episode_rewards(
    rewards: List[List[np.ndarray]], model_names: List[str], save_path: str
) -> None:
    for i in range(len(rewards)):
        rew = list(np.cumsum(rewards[i]))

        target_x = list(range(len(list(rewards[i]))))
        plt.plot(target_x, rew, color=COLORS[i], label=model_names[i])

    plt.legend()

    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Reward Received")

    plt.title("Cumulative Reward Over an Episode", fontweight="bold")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _trimmed_idxs(values: List[Union[int, float]], trim_pct: float):
    num_values = len(values)
    trim_idx = floor(num_values * trim_pct)
    sorted_idx = sorted(range(len(values)), key=lambda k: values[k])
    return sorted_idx[trim_idx:-trim_idx]


def get_average_metrics(
    rewards: List[Union[int, float]],
    num_attacks: List[int],
    attack_pct: List[int],
    trimmed_mean: bool = False,
) -> Tuple[float, float, float]:
    num_episodes = len(rewards)

    if trimmed_mean and num_episodes < 3:
        logging.warning("Not enough episodes given for trimming - ignoring.")
        trimmed_mean = False

    if trimmed_mean:
        idxs = _trimmed_idxs(rewards, 0.1)
        trimmed_rewards = deepcopy(np.array(rewards))[idxs]
        trimmed_num_attacks = deepcopy(np.array(num_attacks))[idxs]
        trimmed_attack_pct = deepcopy(np.array(attack_pct))[idxs]
        num_episodes = len(idxs)
    else:
        trimmed_rewards = rewards
        trimmed_num_attacks = num_attacks
        trimmed_attack_pct = attack_pct

    avg_reward = sum(trimmed_rewards) / num_episodes
    avg_attacks = sum(trimmed_num_attacks) / num_episodes
    avg_perc_attack = (sum(trimmed_attack_pct) / num_episodes) * 100

    return avg_reward, avg_attacks, avg_perc_attack
