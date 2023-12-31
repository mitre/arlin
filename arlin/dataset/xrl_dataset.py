import dataclasses
import logging
import os
import time
from copy import deepcopy
from typing import Dict, List, Tuple, Type

import gymnasium as gym
import numpy as np

from arlin.dataset.collectors import BaseDataCollector, RandomDataCollector
from arlin.dataset.collectors.datapoints import BaseDatapoint


class XRLDataset:
    """Class to store experiences from running a policy in an environment."""

    def __init__(
        self,
        environment: gym.Env,
        collector: BaseDataCollector = RandomDataCollector,
        seed: int = 12345,
    ):
        """Initialize an XRLDataset.

        Args:
            environment (gym.Env): Environment to run the policy in.
            collector (BaseDataCollector, optional): Collector we want to use to collect
                our data. Defaults to RandomDataCollector.
            seed (int, optional): Sed for episode creation. Defaults to 12345.
        """
        self.env = environment
        self.collector = collector
        self.seed = seed

        self.num_datapoints = 0
        self.analyzed = False

        for field in dataclasses.fields(self.collector.datapoint_cls):
            if not hasattr(self, field.name):
                setattr(self, field.name, np.array([], dtype=np.float64))

    def __len__(self) -> int:
        """Number of transitions in the dataset.

        Returns:
            int: Number of transitions in the dataset
        """
        return self.num_datapoints

    def fill(self, num_datapoints: int = 50000, randomness: float = 0.0) -> None:
        """Add transitions to this dataset.

        Args:
            num_datapoints (int, optional): Number of datapoints to add.
                Defaults to 50000.
            randomness (float, optional): How much randomness do we want when taking
                actions. Defaults to 0.0.
        """
        logging.info(f"Collecting {num_datapoints} datapoints.")
        collected_datapoints = 0
        num_episodes = 0
        datapoint_list = []
        self._episode_lens = []
        trunc_count = 0
        while collected_datapoints < num_datapoints:
            datapoints, trunc = self._collect_episode(
                seed=self.seed + num_episodes + trunc_count, randomness=randomness
            )
            if trunc:
                logging.debug("\tSkipping episode due to truncation.")
                trunc_count += 1
                if trunc_count >= 5:
                    err_str = (
                        "Too many truncated episodes in a row identified - "
                        + "please try modifying the randomness value."
                    )
                    raise RuntimeError(err_str)
                continue
            trunc_count = 0
            datapoint_list += datapoints

            collected_datapoints += len(datapoints)
            num_episodes += 1

            logging.info(
                f"\tEpisode {num_episodes} |"
                f" Collected: {len(datapoints)} |"
                f" Total: {collected_datapoints}"
            )

        logging.info(f"Collected {collected_datapoints} datapoints total.")
        if collected_datapoints > num_datapoints:
            num_extra = collected_datapoints - num_datapoints
            logging.debug(
                f"{num_extra} datapoint(s) have been collected for cleaner MDP creation."
            )

        self._append_datapoints(datapoint_list)
        self._analyze_dataset()
        self.num_datapoints += collected_datapoints

    def _collect_episode(
        self, seed: int, randomness: float = 0.0
    ) -> Tuple[List[Type[BaseDatapoint]], bool]:
        """Collect datapoints from a single episode.

        Args:
            seed (int): Seed for the episode.
            randomness (float, optional): How much randomness do we want when taking
                actions. Defaults to 0.0.

        Returns:
            Tuple[List[Type[BaseDatapoint]], bool]: Datapoints, whether this episode was
                truncated or not
        """
        ep_datapoints = []
        obs, _ = self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        step = 0
        render = self.env.render()
        rng = np.random.default_rng(seed)
        term = False
        trunc = False

        while True:
            take_rand_action = rng.random() <= randomness

            if step == 0:
                take_rand_action = False

            if take_rand_action:
                action = self.env.action_space.sample()
            else:
                datapoint, action = self.collector.collect_internal_data(observation=obs)

            new_obs, reward, term, trunc, _ = self.env.step(action)

            datapoint.add_base_data(obs, action, reward, term, trunc, step, render)
            ep_datapoints.append(datapoint)
            render = self.env.render()

            step += 1
            obs = new_obs

            if term or trunc:
                break

        if term:
            self._episode_lens += [step] * len(ep_datapoints)
        return ep_datapoints, trunc

    def _append_datapoints(self, datapoints: List[Type[BaseDatapoint]]):
        """Append the given datapoints to the dataset.

        Args:
            datapoints (List[Type[BaseDatapoint]]): Datapoints to add to the dataset
        """
        start = time.time()
        field_names = [i.name for i in dataclasses.fields(self.collector.datapoint_cls)]

        data_dict = {i: [] for i in field_names}

        for i in range(len(datapoints)):
            datapoint = datapoints[i]

            for field_name in field_names:
                val = getattr(datapoint, field_name)
                data_dict[field_name].append(val)

        for field_name in field_names:
            cur_value = getattr(self, field_name)
            new_data = np.array(data_dict[field_name])

            if cur_value.size == 0:
                setattr(self, field_name, new_data)
            else:
                updated_value = np.concatenate([cur_value, new_data])
                setattr(self, field_name, updated_value)
        end = time.time()

        logging.debug(f"Converting datapoints took {(end - start) / 60} minutes.")

    def _init_analyze(self):
        """Initialize the additional analysis metrics."""
        logging.info("Initializing analytics variables.")
        self.total_rewards = np.array([], dtype=np.float64)
        self.start_indices = np.array([], dtype=np.int8)
        self.term_indices = np.array([], dtype=np.int8)
        self.trunc_indices = np.array([], dtype=np.int8)
        self.unique_state_indices = np.array([], dtype=np.int8)
        self.state_mapping = np.array([], dtype=np.int8)

        self.steps = self.steps.astype("float32")

    def _analyze_dataset(self):
        """Add additional analysis metrics to the dataset that we can't collect."""
        if not self.analyzed:
            self._init_analyze()

        logging.info("Extracting necessary additional data from dataset.")
        self._set_total_rewards()
        self._set_episode_prog_indices()
        self._normalize_steps()
        self._set_distinct_state_data()
        logging.info("Done setting dataset analysis variables.")
        self.analyzed = True

    def _set_total_rewards(self):
        """Add information about the total reward received at each step."""
        logging.info("\tSetting self.total_rewards.")

        total_rewards = []

        cur_total = 0
        for i in range(self.num_datapoints, len(self.rewards)):
            cur_total += self.rewards[i]
            total_rewards.append(cur_total)

            if self.terminateds[i] or self.truncateds[i]:
                cur_total = 0

        self.total_rewards = np.concatenate([self.total_rewards, np.array(total_rewards)])

    def _set_episode_prog_indices(self):
        """Extract episode start and termination indices from the dataset."""

        logging.info("\tSetting self.start_indices.")
        logging.info("\tSetting self.term_indices.")
        logging.info("\tSetting self.trunc_indices.")

        trunc_steps = self.steps[self.num_datapoints : len(self.steps)]
        trunc_terms = self.terminateds[self.num_datapoints : len(self.terminateds)]
        trunc_truncs = self.truncateds[self.num_datapoints : len(self.truncateds)]

        start_indices = np.where(trunc_steps == 0)[0] + self.num_datapoints
        term_indices = np.where(trunc_terms == 1)[0] + self.num_datapoints
        trunc_indices = np.where(trunc_truncs == 1)[0] + self.num_datapoints

        self.start_indices = np.concatenate([self.start_indices, start_indices])
        self.term_indices = np.concatenate([self.term_indices, term_indices])
        self.trunc_indices = np.concatenate([self.trunc_indices, trunc_indices])

        if len(start_indices) == 0:
            logging.warning("No start indices identified.")

        if len(term_indices) == 0:
            logging.warning("No terminated indices identified.")

        if len(trunc_indices) == 0:
            logging.warning("No truncated indices identified.")

    def _normalize_steps(self):
        """Normalize the steps between 0 and 1 depending on time in episode taken."""
        logging.info("\tNormalizing self.steps.")
        # Only get the data from the most recent fill
        cur_fill_steps = deepcopy(self.steps[self.num_datapoints : len(self.steps)])
        normalized_steps = []

        for i in range(len(cur_fill_steps)):
            step = cur_fill_steps[i]
            normalized_steps.append(step / self._episode_lens[i])

        self.steps[self.num_datapoints : len(self.steps)] = normalized_steps

    def _set_distinct_state_data(self):
        """Extract the unique state indices and corresponding state mapping to identify
        unique observations in the dataset. T-SNE has trouble with duplicate states so
        mapping unique states together is beneficial.
        """

        logging.info("\tSetting self.unique_state_indices.")
        logging.info("\tSetting self.state_mapping.")

        outputs = np.unique(
            self.observations, return_index=True, return_inverse=True, axis=0
        )

        _, unique_state_indices, state_mapping = outputs
        self.unique_state_indices = unique_state_indices
        self.state_mapping = state_mapping

    def get_dict(self) -> Dict[str, List[np.ndarray]]:
        """Get a dictionary representation of this dataset.

        Returns:
            Dict[str, List[np.ndarray]]: Dictionary representation of this dataset.
        """
        out_dict = {}

        for field in dataclasses.fields(self.collector.datapoint_cls):
            out_dict[field.name] = np.array(getattr(self, field.name))

        if self.analyzed:
            out_dict["total_rewards"] = self.total_rewards
            out_dict["start_indices"] = self.start_indices
            out_dict["term_indices"] = self.term_indices
            out_dict["trunc_indices"] = self.trunc_indices
            out_dict["unique_state_indices"] = self.unique_state_indices
            out_dict["state_mapping"] = self.state_mapping

        return out_dict

    def save(self, file_path: str) -> None:
        """
        Save dictionary of datapoints to the given file_path.

        Args:
            - file_path str: Filepath to save XRL dataset to.
        """

        if not file_path[-4:] == ".npz":
            file_path += ".npz"

        logging.info(f"Saving datapoints to {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        start = time.time()

        np.savez_compressed(file_path, **self.get_dict())
        end = time.time()

        file_size = round(os.path.getsize(file_path) >> 20, 2)
        logging.debug(f"\tFile size: {file_size} MB")
        logging.debug(f"\tSaved dataset in {(end - start) % 60} minutes.")

    def load(self, load_path: str) -> None:
        """Load a XRLDataset from the given path.

        Args:
            load_path (str): Path to saved XRLDataset.

        Raises:
            ValueError: Missing a required dataset key.
            ValueError: There is no data to load.
            ValueError: Input keys do not have the same number of datapoints.
        """
        dataset = np.load(load_path)

        lens = set()
        for key in [
            "observations",
            "actions",
            "rewards",
            "terminateds",
            "truncateds",
            "steps",
            "renders",
        ]:
            if key not in dataset:
                raise ValueError(f"Invalid dataset - missing {key}.")
            if len(dataset[key]) == 0:
                raise ValueError(f"Key {key} has no associated data.")
            lens.add(len(dataset[key]))

        if len(lens) > 1:
            raise ValueError("Input keys do not have the same number of datapoints.")

        for key in dataset:
            setattr(self, key, dataset[key])

        self.num_datapoints = len(dataset["observations"])

        try:
            getattr(self, "total_rewards")
            self.analyzed = True
        except Exception:
            self.analyzed = False
