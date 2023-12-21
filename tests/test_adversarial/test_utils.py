import os

import numpy as np
import pytest
from PIL import Image

from arlin.adversarial import utils


class TestUtils:
    def test_get_model_name(self):
        name = utils.get_model_name("random", 1, 0)
        assert name == "Random_1"
        name = utils.get_model_name("adversarial", 10, 0)
        assert name == "Adversarial_10"
        name = utils.get_model_name("preference", 1, 0)
        assert name == "Preference_0"
        name = utils.get_model_name("AdVeRSarial", 1, 0)
        assert name == "Adversarial_1"

        with pytest.raises(ValueError):
            utils.get_model_name("adversaria1", 1, 0)

    def test_create_dirs(self, tmpdir):
        utils.create_dirs(tmpdir, ["test1", "test2", "test3"])

        assert os.path.isdir(os.path.join(tmpdir, "gifs"))
        assert os.path.isdir(os.path.join(tmpdir, "metrics"))
        assert os.path.isdir(os.path.join(tmpdir, "metrics", "cosine_similarity"))
        assert os.path.isdir(os.path.join(tmpdir, "metrics", "episode_rewards"))

        assert os.path.isdir(os.path.join(tmpdir, "gifs", "test1"))
        assert os.path.isdir(os.path.join(tmpdir, "gifs", "test2"))
        assert os.path.isdir(os.path.join(tmpdir, "gifs", "test3"))

    def test_save_gifs(self, tmpdir):
        dummy_im = np.zeros([10, 10])
        im = Image.fromarray(dummy_im)
        ep_ims = [im] * 5
        all_ep_ims = [ep_ims] * 5
        rewards = [1, 4, 5, 4, 0]

        utils.save_gifs(all_ep_ims, rewards, tmpdir)

        assert os.path.isfile(os.path.join(tmpdir, "episode_4-min.gif"))
        assert os.path.isfile(os.path.join(tmpdir, "episode_2-max.gif"))

        rewards = [1, 5, 5, 0, 0]
        utils.save_gifs(all_ep_ims, rewards, tmpdir)

        assert os.path.isfile(os.path.join(tmpdir, "episode_3-min.gif"))
        assert os.path.isfile(os.path.join(tmpdir, "episode_1-max.gif"))
