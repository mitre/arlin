from typing import List, Tuple

import gymnasium as gym
import numpy as np
from PIL import Image

from arlin.adversarial.metrics import get_average_metrics, kl_divergence
from arlin.adversarial.utils import get_model_name
from arlin.dataset.collectors import BaseDataCollector


def should_attack(
    model_type: str,
    timestep: int,
    freq: int = 0,
    preference: float = 0,
    threshold: float = 1.0,
) -> bool:
    """Check whether or not we should attack at the given timestep.

    Args:
        model_type (str): Type of model we want to run.
        timestep (int): Current timestep
        freq (int, optional): Frequency of attack. Defaults to 0.
        preference (float, optional): Delta between most and least preferred action.
            Defaults to 0.
        threshold (float, optional): Threshold for preference attack. Defaults to 1.0.

    Raises:
        ValueError: If invalid model type is given.

    Returns:
        bool: Whether or not to attack
    """

    model_type = model_type.lower().capitalize()

    if model_type == "Random" or model_type == "Adversarial":
        return timestep % freq == 0
    elif model_type == "Preference":
        return preference > threshold
    else:
        raise ValueError(f"Model type {model_type} not supported.")


def random_action(env: gym.Env, seed=12345) -> int:
    rng = np.random.default_rng(seed)
    action = rng.integers(low=0, high=env.action_space.n, size=1).item()
    return action


def adversarial_action(obs: np.ndarray, adv_model) -> int:
    action, _ = adv_model.predict(obs, deterministic=True)
    return action


def run_baseline(env: gym.Env, model, num_episodes: int = 10) -> List[np.ndarray]:
    output = {"obs": [], "final_rewards": [], "step_rewards": [], "renders": []}

    # For each eval episode
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=1234 + ep)
        ep_images = [Image.fromarray(env.render())]
        ep_obs = [obs]
        ep_step_rewards = []
        ep_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            ep_images.append(Image.fromarray(env.render()))
            ep_obs.append(obs)
            ep_reward += reward
            ep_step_rewards.append(reward)
            done = terminated or truncated

        output["obs"].append(ep_obs)
        output["final_rewards"].append(ep_reward)
        output["step_rewards"].append(ep_step_rewards)
        output["renders"].append(ep_images)

    avg_reward, _, _ = get_average_metrics(
        rewards=output["final_rewards"],
        num_attacks=[0] * num_episodes,
        attack_pct=[0] * num_episodes,
        trimmed_mean=False,
    )

    print("Model: Baseline")
    print(f"\tAvg Reward: {avg_reward}")

    return output["obs"], output["step_rewards"], output["renders"]


def run_adversarial(
    model_type: str,
    collector: BaseDataCollector,
    env: gym.Env,
    model,
    adv_model,
    attack_freq: int = 0,
    pref_threshold: float = 1.0,
    num_episodes: int = 10,
) -> Tuple[List[np.ndarray], List[float], List[float]]:
    output = {
        "obs": [],
        "final_rewards": [],
        "step_rewards": [],
        "renders": [],
        "num_attacks": [],
        "attack_pct": [],
        "kl_divergence": [],
    }

    model_type = model_type.lower().capitalize()
    model_name = get_model_name(model_type, attack_freq, pref_threshold)

    # For each eval episode
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=1234 + ep)
        ep_images = [Image.fromarray(env.render())]
        ep_obs = [obs]
        ep_step_rewards = []
        num_attacks = 0
        step = 0
        ep_reward = 0
        done = False

        while not done:
            internal_data, _ = collector.collect_internal_data(obs)
            probs = internal_data.dist_probs
            preference = probs.max() - probs.min()

            if should_attack(model_type, step, attack_freq, preference, pref_threshold):
                if model_type == "Random":
                    action = random_action(env)
                elif model_type in ["Adversarial", "Preference"]:
                    action = adversarial_action(obs, adv_model)
                else:
                    raise ValueError(f"Invalid model_type {model_type} given.")
                num_attacks += 1

                output["kl_divergence"].append(kl_divergence(probs, action))
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)

            ep_images.append(Image.fromarray(env.render()))
            ep_obs.append(obs)
            ep_reward += reward
            ep_step_rewards.append(reward)
            done = terminated or truncated
            step += 1

        output["obs"].append(ep_obs)
        output["final_rewards"].append(ep_reward)
        output["step_rewards"].append(ep_step_rewards)
        output["renders"].append(ep_images)
        output["num_attacks"].append(num_attacks)
        output["attack_pct"].append(num_attacks / step)

    (avg_reward, avg_attacks, avg_attack_pct) = get_average_metrics(
        rewards=output["final_rewards"],
        num_attacks=output["num_attacks"],
        attack_pct=output["attack_pct"],
        trimmed_mean=True,
    )

    print(f"Model: {model_name}")
    print(
        f"""\tAvg Reward: {avg_reward} |
        Avg Num Attacks: {avg_attacks} |
        Avg Percent Attacks: {avg_attack_pct}"""
    )

    return (
        output["obs"],
        output["step_rewards"],
        output["kl_divergence"],
        output["renders"],
    )


def run_arlin(
    collector: BaseDataCollector,
    env: gym.Env,
    model,
    start_algo,
    mid_algo,
    term_algo,
    num_episodes: int = 10,
) -> Tuple[List[np.ndarray], List[float], List[float]]:
    output = {
        "obs": [],
        "final_rewards": [],
        "step_rewards": [],
        "renders": [],
        "num_attacks": [],
        "attack_pct": [],
        "kl_divergence": [],
        "term_clusters": [],
    }

    # For each eval episode
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=1234 + ep)
        ep_images = [Image.fromarray(env.render())]
        ep_obs = [obs]
        ep_step_rewards = []
        num_attacks = 0
        step = 0
        ep_reward = 0
        done = False

        while not done:
            internal_data, _ = collector.collect_internal_data(obs)
            probs = internal_data.dist_probs
            latent = internal_data.latent_actors
            value = internal_data.critic_values

            data = np.concatenate([latent, np.expand_dims(value, axis=-1)], axis=-1)

            if step == 0:
                prediction = start_algo.predict(data.reshape(1, -1)) + 20
            else:
                prediction = mid_algo.predict(data.reshape(1, -1))

            if prediction == 1:
                action = 2
                num_attacks += 1
                div = kl_divergence(probs, action)
                output["kl_divergence"].append(div)
            elif prediction in [7, 5]:
                action = 3
                num_attacks += 1
                div = kl_divergence(probs, action)
                output["kl_divergence"].append(div)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)

            ep_images.append(Image.fromarray(env.render()))
            ep_obs.append(obs)
            ep_reward += reward
            ep_step_rewards.append(reward)
            done = terminated or truncated
            step += 1

            if done:
                data = np.concatenate(
                    [
                        latent,
                        np.expand_dims(value, axis=-1),
                        np.expand_dims(reward, axis=-1),
                    ],
                    axis=-1,
                )
                term_pred = term_algo.predict(data.reshape(1, -1)) + 22

        output["obs"].append(ep_obs)
        output["final_rewards"].append(ep_reward)
        output["step_rewards"].append(ep_step_rewards)
        output["renders"].append(ep_images)
        output["num_attacks"].append(num_attacks)
        output["attack_pct"].append(num_attacks / step)
        output["term_clusters"].append(term_pred)

    (avg_reward, avg_attacks, avg_attack_pct) = get_average_metrics(
        rewards=output["final_rewards"],
        num_attacks=output["num_attacks"],
        attack_pct=output["attack_pct"],
        trimmed_mean=True,
    )

    print("Model: ARLIN")
    print(
        f"""\tAvg Reward: {avg_reward} |
        Avg Num Attacks: {avg_attacks} |
        Avg Percent Attacks: {avg_attack_pct}"""
    )
    reached_pct = int(
        (sum([i == 23 for i in output["term_clusters"]]) / num_episodes) * 100
    )
    print(f"\tReached target cluster in {reached_pct}% of trials")

    return (
        output["obs"],
        output["step_rewards"],
        output["kl_divergence"],
        output["renders"],
    )
