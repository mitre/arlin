import dataclasses
import os

import gymnasium as gym
import numpy as np
import pytest

from arlin.dataset import XRLDataset
from arlin.dataset.collectors import RandomDataCollector, SB3PPODataCollector
from arlin.dataset.collectors.datapoints import BaseDatapoint, SB3PPODatapoint
from arlin.dataset.loaders import load_hf_sb_model


@pytest.fixture
def env():
    # Create environment
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    return env


@pytest.fixture
def collector(env):
    # Create the datapoint collector for SB3 PPO Datapoints with the model's policy
    collector = RandomDataCollector(datapoint_cls=BaseDatapoint, environment=env)
    return collector


@pytest.fixture
def ppo_collector():
    # Load the SB3 model from Huggingface
    model = load_hf_sb_model(
        repo_id="sb3/ppo-LunarLander-v2",
        filename="ppo-LunarLander-v2.zip",
        algo_str="ppo",
    )

    # Create the datapoint collector for SB3 PPO Datapoints with the model's policy
    collector = SB3PPODataCollector(datapoint_cls=SB3PPODatapoint, policy=model.policy)
    return collector


@pytest.fixture
def dataset(collector, env):
    # Instantiate the XRL Dataset
    dataset = XRLDataset(env, collector=collector)
    return dataset


@pytest.fixture
def ppo_dataset(ppo_collector, env):
    # Instantiate the XRL Dataset
    dataset = XRLDataset(env, collector=ppo_collector)
    return dataset


class TestXRLDataset:
    def test_init(self, collector, env):
        dataset = XRLDataset(env, collector=collector)

        assert dataset.env == env
        assert dataset.collector == collector

        assert dataset.num_datapoints == 0
        assert not dataset.analyzed

        for field in dataclasses.fields(dataset.collector.datapoint_cls):
            assert np.array_equal(
                getattr(dataset, field.name), np.array([], dtype=np.float64)
            )

    def test_fill(self, dataset, ppo_dataset):
        assert dataset.num_datapoints == 0
        dataset.fill(num_datapoints=500, randomness=0.25)
        old_val = dataset.num_datapoints
        assert dataset.num_datapoints >= 500
        assert dataset.analyzed

        assert len(dataset.observations) == dataset.num_datapoints
        assert len(dataset.actions) == dataset.num_datapoints
        assert len(dataset.rewards) == dataset.num_datapoints
        assert len(dataset.terminateds) == dataset.num_datapoints
        assert len(dataset.truncateds) == dataset.num_datapoints
        assert len(dataset.steps) == dataset.num_datapoints
        assert len(dataset.renders) == dataset.num_datapoints
        assert len(dataset.total_rewards) == dataset.num_datapoints
        assert dataset.start_indices is not None
        assert dataset.term_indices is not None
        assert dataset.trunc_indices is not None
        assert dataset.unique_state_indices is not None
        assert dataset.state_mapping is not None

        dataset.fill(num_datapoints=500, randomness=0.25)
        assert dataset.num_datapoints > (old_val + 500)

        assert len(dataset.observations) == dataset.num_datapoints
        assert len(dataset.actions) == dataset.num_datapoints
        assert len(dataset.rewards) == dataset.num_datapoints
        assert len(dataset.terminateds) == dataset.num_datapoints
        assert len(dataset.truncateds) == dataset.num_datapoints
        assert len(dataset.steps) == dataset.num_datapoints
        assert len(dataset.renders) == dataset.num_datapoints
        assert len(dataset.total_rewards) == dataset.num_datapoints
        assert len(dataset.start_indices) > 0
        assert len(dataset.term_indices) > 0
        assert dataset.trunc_indices is not None
        assert dataset.unique_state_indices is not None
        assert dataset.state_mapping is not None

    def test_collect_episode(self, dataset):
        dataset._episode_lens = []
        datapoints_1, truncated_1 = dataset._collect_episode(1234)

        datapoints_2, truncated_2 = dataset._collect_episode(1234)

        for dp1, dp2 in zip(datapoints_1, datapoints_2):
            assert dp1 == dp2
        assert truncated_1 == truncated_2

    def test_append_datapoints(self, dataset):
        dataset._episode_lens = []
        datapoints, _ = dataset._collect_episode(1234)

        assert dataset.num_datapoints == 0
        num_dp = len(datapoints)
        dataset._append_datapoints(datapoints)
        assert len(dataset.observations) == num_dp
        assert len(dataset.actions) == num_dp
        assert len(dataset.rewards) == num_dp
        assert len(dataset.terminateds) == num_dp
        assert len(dataset.truncateds) == num_dp
        assert len(dataset.steps) == num_dp
        assert len(dataset.renders) == num_dp

    def test_init_analyze(self, dataset):
        assert not dataset.analyzed

        with pytest.raises(Exception):
            dataset.total_rewards

        with pytest.raises(Exception):
            dataset.start_indices

        with pytest.raises(Exception):
            dataset.term_indices

        with pytest.raises(Exception):
            dataset.trunc_indices

        with pytest.raises(Exception):
            dataset.unique_state_indices

        with pytest.raises(Exception):
            dataset.state_mapping

        dataset._init_analyze()
        assert np.array_equal(dataset.total_rewards, np.array([], dtype=np.float64))
        assert dataset.total_rewards.dtype == np.float64
        assert np.array_equal(dataset.start_indices, np.array([], dtype=np.int8))
        assert dataset.start_indices.dtype == np.int8
        assert np.array_equal(dataset.term_indices, np.array([], dtype=np.int8))
        assert dataset.term_indices.dtype == np.int8
        assert np.array_equal(dataset.trunc_indices, np.array([], dtype=np.int8))
        assert dataset.trunc_indices.dtype == np.int8
        assert np.array_equal(dataset.unique_state_indices, np.array([], dtype=np.int8))
        assert dataset.unique_state_indices.dtype == np.int8
        assert np.array_equal(dataset.state_mapping, np.array([], dtype=np.int8))
        assert dataset.state_mapping.dtype == np.int8
        assert dataset.steps.dtype == np.float32

    def test_analyze_dataset(self, dataset):
        assert not dataset.analyzed

        dataset._episode_lens = []
        datapoints, _ = dataset._collect_episode(1234)
        dataset._append_datapoints(datapoints)

        dataset._analyze_dataset()
        assert dataset.analyzed

    def test_set_total_rewards(self, dataset):
        with pytest.raises(Exception):
            dataset.total_rewards

        dataset._episode_lens = []
        datapoints, _ = dataset._collect_episode(1234)
        num_dp = len(datapoints)
        dataset._append_datapoints(datapoints)
        dataset._init_analyze()
        dataset._set_total_rewards()
        assert len(dataset.total_rewards) == num_dp

        for i in range(1, len(dataset.total_rewards)):
            test_val = dataset.total_rewards[i - 1] + dataset.rewards[i]
            assert dataset.total_rewards[i] == test_val

    def test_set_episode_prog(self, dataset):
        with pytest.raises(Exception):
            dataset.start_indices

        with pytest.raises(Exception):
            dataset.term_indices

        with pytest.raises(Exception):
            dataset.trunc_indices

        dataset._episode_lens = []
        datapoints, _ = dataset._collect_episode(1234)
        assert len(datapoints) > 0
        dataset._append_datapoints(datapoints)
        dataset._init_analyze()
        dataset._set_episode_prog_indices()

        num_starts = len(np.where(dataset.steps == 0)[0])
        num_terms = len(np.where(dataset.terminateds == 1)[0])
        num_truncs = len(np.where(dataset.truncateds == 1)[0])

        assert num_starts >= 1
        assert num_terms + num_truncs >= 1

        assert len(dataset.start_indices) == num_starts
        assert len(dataset.term_indices) == num_terms
        assert len(dataset.trunc_indices) == num_truncs

    def test_normalize_steps(self, dataset):
        dataset._episode_lens = []
        datapoints, _ = dataset._collect_episode(1234)
        dataset._append_datapoints(datapoints)
        assert max(dataset.steps) > 1
        dataset._normalize_steps()

        assert max(dataset.steps <= 1.0)
        assert min(dataset.steps >= 0.0)

    def test_set_distinct_state_data(self, dataset):
        with pytest.raises(Exception):
            dataset.unique_state_indices

        with pytest.raises(Exception):
            dataset.state_mapping

        dataset._episode_lens = []
        datapoints, _ = dataset._collect_episode(1234)
        dataset._append_datapoints(datapoints)
        dataset._init_analyze()
        dataset._set_distinct_state_data()

        assert len(dataset.state_mapping) == len(dataset.observations)

    def test_get_dict(self, dataset):
        dataset.fill(num_datapoints=50, randomness=0.25)
        output = dataset.get_dict()
        output_keys = list(output.keys())

        for field in dataclasses.fields(dataset.collector.datapoint_cls):
            assert field.name in output_keys
            assert np.array_equal(output[field.name], getattr(dataset, field.name))
            assert output[field.name].dtype == getattr(dataset, field.name).dtype

        for field in [
            "total_rewards",
            "start_indices",
            "term_indices",
            "trunc_indices",
            "unique_state_indices",
            "state_mapping",
        ]:
            assert field in output_keys
            assert np.array_equal(output[field], getattr(dataset, field))
            assert output[field].dtype == getattr(dataset, field).dtype

    def test_save_load(self, dataset, collector, env, tmpdir):
        test_path = os.path.join(tmpdir, "dataset_test.npz")
        dataset.save(test_path)
        with pytest.raises(ValueError):
            dataset.load(test_path)

        dataset.fill(num_datapoints=50, randomness=0.25)
        dataset_path_1 = os.path.join(tmpdir, "dataset.npz")
        dataset_path_2 = os.path.join(tmpdir, "dataset_2")
        dataset.save(dataset_path_1)
        assert os.path.isfile(dataset_path_1)
        dataset.save(dataset_path_2)
        assert os.path.isfile(dataset_path_2 + ".npz")

        loaded_dataset = XRLDataset(env, collector=collector)
        assert loaded_dataset.num_datapoints == 0
        assert not loaded_dataset.analyzed

        for field in dataclasses.fields(loaded_dataset.collector.datapoint_cls):
            loaded_field = getattr(loaded_dataset, field.name)
            assert np.array_equal(loaded_field, np.array([], dtype=np.float64))
            assert getattr(loaded_dataset, field.name).dtype == np.float64

        loaded_dataset.load(dataset_path_1)
        assert loaded_dataset.num_datapoints == dataset.num_datapoints
        assert loaded_dataset.analyzed == dataset.analyzed

        for field in dataclasses.fields(loaded_dataset.collector.datapoint_cls):
            loaded_field = getattr(loaded_dataset, field.name)
            dataset_field = getattr(dataset, field.name)
            assert np.array_equal(loaded_field, dataset_field)
            assert loaded_field.dtype == dataset_field.dtype
