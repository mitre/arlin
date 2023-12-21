import os

import numpy as np

from arlin.adversarial import metrics


class TestMetrics:
    def test_calculate_cosine_sim(self):
        base = [[1, 1], [1, 1], [1, 1]]
        target = [[1, 1], [1, 1], [1, 1]]
        xs, ys = metrics._calculate_cosine_sim(base, target)

        assert xs == [0, 1, 2]
        assert ys == [1, 1, 1]

        base = [[1, 1], [1, 1], [1, 1]]
        target = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
        xs, ys = metrics._calculate_cosine_sim(base, target)

        assert xs == [0, 1, 2, 3, 4]
        assert ys == [1, 1, 1, 1, 1]

        base = [[1, 1], [1, 1], [1, 1]]
        target = [[1, 1], [1, 1], [1, 1], [2, 1], [2, 1]]
        xs, ys = metrics._calculate_cosine_sim(base, target)

        assert xs == [0, 1, 2, 3, 4]
        assert ys == [1, 1, 1, 0.948683, 0.948683]

    def test_plot_cosine_sim(self, tmpdir):
        base = [[1, 1], [1, 1], [1, 1]]
        target = [[[1, 1], [1, 1], [1, 1]]]

        metrics.plot_cosine_sim(base, target, ["test"], os.path.join(tmpdir, "test.png"))

        os.path.isfile(os.path.join(tmpdir, "test.png"))

    def test_kl_divergence(self):
        dist = np.array([0, 0.2, 0.5, 0.5])
        adv = 3

        div = metrics.kl_divergence(dist, adv)
        assert div == 0

        dist = np.array([0, 0.2, 0.5, 0.3])
        div = metrics.kl_divergence(dist, adv)
        assert div == 0.025541281188299528

    def test_mx_x(self):
        stats = [[1, 0.5], [3, 0.25], [0, 0.1]]

        max_x = metrics._max_x(stats)

        assert max_x == 3.5

    def test_plot_divergences(self, tmpdir):
        test = [0.5, 0.1, 0.3, 0.4]
        metrics.plot_divergences([test], ["test"], os.path.join(tmpdir, "test.png"))

        os.path.isfile(os.path.join(tmpdir, "test.png"))

    def test_plot_episode_rewards(self, tmpdir):
        rewards = np.array([1, 1, 1, 1, 1, 1])
        metrics.plot_episode_rewards(
            [rewards], ["test"], os.path.join(tmpdir, "test.png")
        )

        os.path.isfile(os.path.join(tmpdir, "test.png"))

    def test_trimmed_idxs(self):
        values = [1, 2, 3, 4, 0, 1, 4, 2, 3, 7]

        new_idxs = metrics._trimmed_idxs(values, 0.1)
        assert new_idxs == [0, 5, 1, 7, 2, 8, 3, 6]

    def test_get_average_metrics(self):
        rewards = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        num_attacks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        attack_pct = [0.10, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        avg_rew, avg_attacks, avg_perc_attack = metrics.get_average_metrics(
            rewards, num_attacks, attack_pct, True
        )

        assert avg_rew == 27.5
        assert avg_attacks == 5.5
        assert avg_perc_attack == 55.00
