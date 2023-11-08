import dataclasses
import os

import numpy as np
import pytest

from arlin.dataset import XRLDataset
from arlin.dataset.collectors import RandomDataCollector
from arlin.dataset.collectors.datapoints import BaseDatapoint


@pytest.fixture
def uf_random_dataset(env):
    # Create the datapoint collector for SB3 PPO Datapoints with the model's policy
    collector = RandomDataCollector(datapoint_cls=BaseDatapoint, environment=env)
    # Instantiate the XRL Dataset
    dataset = XRLDataset(env, collector=collector)

    return dataset


@pytest.fixture
def random_collector(env):
    # Create the datapoint collector for SB3 PPO Datapoints with the model's policy
    collector = RandomDataCollector(datapoint_cls=BaseDatapoint, environment=env)
    return collector


class TestXRLDataset:
    def test_init(self, random_collector, env):
        dataset = XRLDataset(env, collector=random_collector)

        assert dataset.env == env
        assert dataset.collector == random_collector

        assert dataset.num_datapoints == 0
        assert not dataset.analyzed

        for field in dataclasses.fields(dataset.collector.datapoint_cls):
            assert np.array_equal(
                getattr(dataset, field.name), np.array([], dtype=np.float64)
            )

    def test_fill(self, uf_random_dataset):
        assert uf_random_dataset.num_datapoints == 0
        uf_random_dataset.fill(num_datapoints=500, randomness=0.25)
        old_val = uf_random_dataset.num_datapoints
        assert uf_random_dataset.num_datapoints >= 500
        assert uf_random_dataset.analyzed

        assert len(uf_random_dataset.observations) == uf_random_dataset.num_datapoints
        assert len(uf_random_dataset.actions) == uf_random_dataset.num_datapoints
        assert len(uf_random_dataset.rewards) == uf_random_dataset.num_datapoints
        assert len(uf_random_dataset.terminateds) == uf_random_dataset.num_datapoints
        assert len(uf_random_dataset.truncateds) == uf_random_dataset.num_datapoints
        assert len(uf_random_dataset.steps) == uf_random_dataset.num_datapoints
        assert len(uf_random_dataset.renders) == uf_random_dataset.num_datapoints
        assert len(uf_random_dataset.total_rewards) == uf_random_dataset.num_datapoints
        assert uf_random_dataset.start_indices is not None
        assert uf_random_dataset.term_indices is not None
        assert uf_random_dataset.trunc_indices is not None
        assert uf_random_dataset.unique_state_indices is not None
        assert uf_random_dataset.state_mapping is not None

        uf_random_dataset.fill(num_datapoints=500, randomness=0.25)
        assert uf_random_dataset.num_datapoints > (old_val + 500)

        assert len(uf_random_dataset.observations) == uf_random_dataset.num_datapoints
        assert len(uf_random_dataset.actions) == uf_random_dataset.num_datapoints
        assert len(uf_random_dataset.rewards) == uf_random_dataset.num_datapoints
        assert len(uf_random_dataset.terminateds) == uf_random_dataset.num_datapoints
        assert len(uf_random_dataset.truncateds) == uf_random_dataset.num_datapoints
        assert len(uf_random_dataset.steps) == uf_random_dataset.num_datapoints
        assert len(uf_random_dataset.renders) == uf_random_dataset.num_datapoints
        assert len(uf_random_dataset.total_rewards) == uf_random_dataset.num_datapoints
        assert len(uf_random_dataset.start_indices) > 0
        assert len(uf_random_dataset.term_indices) > 0
        assert uf_random_dataset.trunc_indices is not None
        assert uf_random_dataset.unique_state_indices is not None
        assert uf_random_dataset.state_mapping is not None

    def test_collect_episode(self, uf_random_dataset):
        uf_random_dataset._episode_lens = []
        datapoints_1, truncated_1 = uf_random_dataset._collect_episode(1234)

        datapoints_2, truncated_2 = uf_random_dataset._collect_episode(1234)

        for dp1, dp2 in zip(datapoints_1, datapoints_2):
            assert dp1 == dp2
        assert truncated_1 == truncated_2

    def test_append_datapoints(self, uf_random_dataset):
        uf_random_dataset._episode_lens = []
        datapoints, _ = uf_random_dataset._collect_episode(1234)

        assert uf_random_dataset.num_datapoints == 0
        num_dp = len(datapoints)
        uf_random_dataset._append_datapoints(datapoints)
        assert len(uf_random_dataset.observations) == num_dp
        assert len(uf_random_dataset.actions) == num_dp
        assert len(uf_random_dataset.rewards) == num_dp
        assert len(uf_random_dataset.terminateds) == num_dp
        assert len(uf_random_dataset.truncateds) == num_dp
        assert len(uf_random_dataset.steps) == num_dp
        assert len(uf_random_dataset.renders) == num_dp

    def test_init_analyze(self, uf_random_dataset):
        assert not uf_random_dataset.analyzed

        with pytest.raises(Exception):
            uf_random_dataset.total_rewards

        with pytest.raises(Exception):
            uf_random_dataset.start_indices

        with pytest.raises(Exception):
            uf_random_dataset.term_indices

        with pytest.raises(Exception):
            uf_random_dataset.trunc_indices

        with pytest.raises(Exception):
            uf_random_dataset.unique_state_indices

        with pytest.raises(Exception):
            uf_random_dataset.state_mapping

        uf_random_dataset._init_analyze()
        assert np.array_equal(
            uf_random_dataset.total_rewards, np.array([], dtype=np.float64)
        )
        assert uf_random_dataset.total_rewards.dtype == np.float64
        assert np.array_equal(
            uf_random_dataset.start_indices, np.array([], dtype=np.int8)
        )
        assert uf_random_dataset.start_indices.dtype == np.int8
        assert np.array_equal(uf_random_dataset.term_indices, np.array([], dtype=np.int8))
        assert uf_random_dataset.term_indices.dtype == np.int8
        assert np.array_equal(
            uf_random_dataset.trunc_indices, np.array([], dtype=np.int8)
        )
        assert uf_random_dataset.trunc_indices.dtype == np.int8
        assert np.array_equal(
            uf_random_dataset.unique_state_indices, np.array([], dtype=np.int8)
        )
        assert uf_random_dataset.unique_state_indices.dtype == np.int8
        assert np.array_equal(
            uf_random_dataset.state_mapping, np.array([], dtype=np.int8)
        )
        assert uf_random_dataset.state_mapping.dtype == np.int8
        assert uf_random_dataset.steps.dtype == np.float32

    def test_analyze_dataset(self, uf_random_dataset):
        assert not uf_random_dataset.analyzed

        uf_random_dataset._episode_lens = []
        datapoints, _ = uf_random_dataset._collect_episode(1234)
        uf_random_dataset._append_datapoints(datapoints)

        uf_random_dataset._analyze_dataset()
        assert uf_random_dataset.analyzed

    def test_set_total_rewards(self, uf_random_dataset):
        with pytest.raises(Exception):
            uf_random_dataset.total_rewards

        uf_random_dataset._episode_lens = []
        datapoints, _ = uf_random_dataset._collect_episode(1234)
        num_dp = len(datapoints)
        uf_random_dataset._append_datapoints(datapoints)
        uf_random_dataset._init_analyze()
        uf_random_dataset._set_total_rewards()
        assert len(uf_random_dataset.total_rewards) == num_dp

        for i in range(1, len(uf_random_dataset.total_rewards)):
            test_val = (
                uf_random_dataset.total_rewards[i - 1] + uf_random_dataset.rewards[i]
            )
            assert uf_random_dataset.total_rewards[i] == test_val

    def test_set_episode_prog(self, uf_random_dataset):
        with pytest.raises(Exception):
            uf_random_dataset.start_indices

        with pytest.raises(Exception):
            uf_random_dataset.term_indices

        with pytest.raises(Exception):
            uf_random_dataset.trunc_indices

        uf_random_dataset._episode_lens = []
        datapoints, _ = uf_random_dataset._collect_episode(1234)
        assert len(datapoints) > 0
        uf_random_dataset._append_datapoints(datapoints)
        uf_random_dataset._init_analyze()
        uf_random_dataset._set_episode_prog_indices()

        num_starts = len(np.where(uf_random_dataset.steps == 0)[0])
        num_terms = len(np.where(uf_random_dataset.terminateds == 1)[0])
        num_truncs = len(np.where(uf_random_dataset.truncateds == 1)[0])

        assert num_starts >= 1
        assert num_terms + num_truncs >= 1

        assert len(uf_random_dataset.start_indices) == num_starts
        assert len(uf_random_dataset.term_indices) == num_terms
        assert len(uf_random_dataset.trunc_indices) == num_truncs

    def test_normalize_steps(self, uf_random_dataset):
        uf_random_dataset._episode_lens = []
        datapoints, _ = uf_random_dataset._collect_episode(1234)
        uf_random_dataset._append_datapoints(datapoints)
        assert max(uf_random_dataset.steps) > 1
        uf_random_dataset._normalize_steps()

        assert max(uf_random_dataset.steps <= 1.0)
        assert min(uf_random_dataset.steps >= 0.0)

    def test_set_distinct_state_data(self, uf_random_dataset):
        with pytest.raises(Exception):
            uf_random_dataset.unique_state_indices

        with pytest.raises(Exception):
            uf_random_dataset.state_mapping

        uf_random_dataset._episode_lens = []
        datapoints, _ = uf_random_dataset._collect_episode(1234)
        uf_random_dataset._append_datapoints(datapoints)
        uf_random_dataset._init_analyze()
        uf_random_dataset._set_distinct_state_data()

        assert len(uf_random_dataset.state_mapping) == len(uf_random_dataset.observations)

    def test_get_dict(self, uf_random_dataset):
        uf_random_dataset.fill(num_datapoints=50, randomness=0.25)
        output = uf_random_dataset.get_dict()
        output_keys = list(output.keys())

        for field in dataclasses.fields(uf_random_dataset.collector.datapoint_cls):
            assert field.name in output_keys
            assert np.array_equal(
                output[field.name], getattr(uf_random_dataset, field.name)
            )
            assert (
                output[field.name].dtype == getattr(uf_random_dataset, field.name).dtype
            )

        for field in [
            "total_rewards",
            "start_indices",
            "term_indices",
            "trunc_indices",
            "unique_state_indices",
            "state_mapping",
        ]:
            assert field in output_keys
            assert np.array_equal(output[field], getattr(uf_random_dataset, field))
            assert output[field].dtype == getattr(uf_random_dataset, field).dtype

    def test_save_load(self, uf_random_dataset, random_collector, env, tmpdir):
        test_path = os.path.join(tmpdir, "dataset_test.npz")
        uf_random_dataset.save(test_path)
        with pytest.raises(ValueError):
            uf_random_dataset.load(test_path)

        uf_random_dataset.fill(num_datapoints=50, randomness=0.25)
        dataset_path_1 = os.path.join(tmpdir, "uf_random_dataset.npz")
        dataset_path_2 = os.path.join(tmpdir, "dataset_2")
        uf_random_dataset.save(dataset_path_1)
        assert os.path.isfile(dataset_path_1)
        uf_random_dataset.save(dataset_path_2)
        assert os.path.isfile(dataset_path_2 + ".npz")

        loaded_dataset = XRLDataset(env, collector=random_collector)
        assert loaded_dataset.num_datapoints == 0
        assert not loaded_dataset.analyzed

        for field in dataclasses.fields(loaded_dataset.collector.datapoint_cls):
            loaded_field = getattr(loaded_dataset, field.name)
            assert np.array_equal(loaded_field, np.array([], dtype=np.float64))
            assert getattr(loaded_dataset, field.name).dtype == np.float64

        loaded_dataset.load(dataset_path_1)
        assert loaded_dataset.num_datapoints == uf_random_dataset.num_datapoints
        assert loaded_dataset.analyzed == uf_random_dataset.analyzed

        for field in dataclasses.fields(loaded_dataset.collector.datapoint_cls):
            loaded_field = getattr(loaded_dataset, field.name)
            dataset_field = getattr(uf_random_dataset, field.name)
            assert np.array_equal(loaded_field, dataset_field)
            assert loaded_field.dtype == dataset_field.dtype
