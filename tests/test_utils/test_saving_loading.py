import os

import numpy as np

from arlin.utils.saving_loading import load_data, save_data


class TestSavingLoading:
    def test_save_data(self, tmpdir, ppo_embeddings):
        embeddings_path = os.path.join(tmpdir, "embeddings.npy")
        embeddings_2_path = os.path.join(tmpdir, "tests", "embeddings_2.npy")
        save_data(ppo_embeddings, os.path.join(tmpdir, "embeddings"))
        save_data(ppo_embeddings, os.path.join(tmpdir, "tests", "embeddings_2.npy"))

        assert os.path.isfile(embeddings_path)
        assert os.path.isfile(embeddings_2_path)

    def test_load_data(self, tmpdir, ppo_embeddings):
        embeddings_path = os.path.join(tmpdir, "embeddings.npy")
        save_data(ppo_embeddings, embeddings_path)
        loaded_embeddings = load_data(embeddings_path)

        assert np.array_equal(ppo_embeddings, loaded_embeddings)
