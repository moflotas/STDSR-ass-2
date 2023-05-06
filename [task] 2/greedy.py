import random
import time


class Sampler:
    def __init__(self, reward_functions: list[callable]):
        self.reward_functions = reward_functions
        self.reset()

    def make_iter(self):
        raise NotImplementedError("'make_iter' method is not implemented")

    def make_iters(self, number_of_iters: int):
        for _ in range(number_of_iters):
            self.make_iter()

    def get_stats(self) -> tuple[int, list[int], list[int]]:
        raise NotImplementedError("'get_stats' method is not implemented")

    def reset(self):
        raise NotImplementedError("'reset' method is not implemented")


class Greedy(Sampler):
    def __init__(self, reward_functions: list[callable]):
        super().__init__(reward_functions)

    def make_iter(self):
        optimal_bandit = sorted([i for i in range(len(
            self.reward_functions))], key=lambda x: self.successes[x] / self.total_times[x])[-1]

        self.successes[optimal_bandit] += self.reward_functions[optimal_bandit]()
        self.total_times[optimal_bandit] += 1

        self.number_of_iters += 1

    def get_stats(self) -> tuple[int, list[int], list[int]]:
        return [i / j for i, j in zip(self.successes, self.total_times)]

    def reset(self):
        self.number_of_iters = 0
        self.successes = [1 for _ in range(len(self.reward_functions))]
        self.total_times = [1 for _ in range(len(self.reward_functions))]


class Thompson(Sampler):
    def __init__(self, reward_functions: list[callable]):
        super().__init__(reward_functions)

    def make_iter(self):
        optimal_bandit = sorted([i for i in range(len(self.reward_functions))],
                                key=lambda x: random.betavariate(self.alphas[x], self.betas[x]))[-1]

        result = self.reward_functions[optimal_bandit]()

        self.alphas[optimal_bandit] += result
        self.betas[optimal_bandit] += 1 - result

        self.number_of_iters += 1

    def get_stats(self) -> tuple[int, list[int], list[int]]:
        return [i / (i + j) for i, j in zip(self.alphas, self.betas)]

    def reset(self):
        self.number_of_iters = 0
        self.alphas = [1 for _ in range(len(self.reward_functions))]
        self.betas = [1 for _ in range(len(self.reward_functions))]


class GaussianGreedy(Sampler):
    def __init__(self, reward_functions: list[callable], actual_std: float):
        super().__init__(reward_functions)
        self.actual_std = actual_std

    def make_iter(self):
        optimal_bandit = sorted([i for i in range(len(self.reward_functions))],
                                key=lambda x: self.prior_means[x])[-1]

        self.total_times[optimal_bandit] += 1
        self.cum_rewards[optimal_bandit] += self.reward_functions[optimal_bandit]()

        prev_mu = self.prior_means[optimal_bandit]
        prev_std = self.prior_stds[optimal_bandit]
        cum_reward = self.cum_rewards[optimal_bandit]
        n = self.total_times[optimal_bandit]

        new_variance = 1 / (1 / prev_std ** 2 + n / self.actual_std ** 2)
        new_std = new_variance ** 0.5
        new_mean = new_variance * (prev_mu / prev_std ** 2 + cum_reward / self.actual_std ** 2)
        
        self.prior_means[optimal_bandit] = new_mean
        self.prior_stds[optimal_bandit] = new_std

        self.number_of_iters += 1

    def get_stats(self) -> tuple[int, list[int], list[int]]:
        return self.prior_means, self.cum_rewards, self.total_times

    def reset(self):
        self.number_of_iters = 0
        self.prior_means = [0.5 for _ in range(len(self.reward_functions))]
        self.prior_stds = [0.5 for _ in range(len(self.reward_functions))]
        self.cum_rewards = [0 for _ in range(len(self.reward_functions))]
        self.total_times = [0 for _ in range(len(self.reward_functions))]


class GaussianThompson(Sampler):
    def __init__(self, reward_functions: list[callable], actual_std: float):
        super().__init__(reward_functions)
        self.actual_std = actual_std

    def make_iter(self):
        optimal_bandit = sorted([i for i in range(len(self.reward_functions))],
                                key=lambda x: random.normalvariate(self.prior_means[x], (self.prior_stds[x] ** 2 + self.actual_std ** 2) ** 0.5))[-1]

        self.total_times[optimal_bandit] += 1
        self.cum_rewards[optimal_bandit] += self.reward_functions[optimal_bandit]()

        prev_mu = self.prior_means[optimal_bandit]
        prev_std = self.prior_stds[optimal_bandit]
        cum_reward = self.cum_rewards[optimal_bandit]
        n = self.total_times[optimal_bandit]

        new_variance = 1 / (1 / prev_std ** 2 + n / self.actual_std ** 2)
        new_std = new_variance ** 0.5
        new_mean = new_variance * (prev_mu / prev_std ** 2 + cum_reward / self.actual_std ** 2)
        
        self.prior_means[optimal_bandit] = new_mean
        self.prior_stds[optimal_bandit] = new_std

        self.number_of_iters += 1

    def get_stats(self) -> tuple[int, list[int], list[int]]:
        return self.prior_means

    def reset(self):
        self.number_of_iters = 0
        self.prior_means = [0.5 for _ in range(len(self.reward_functions))]
        self.prior_stds = [0.5 for _ in range(len(self.reward_functions))]
        self.cum_rewards = [0 for _ in range(len(self.reward_functions))]
        self.total_times = [0 for _ in range(len(self.reward_functions))]


if __name__ == '__main__':
    start = time.time()

    # test = Greedy([
    #     lambda: random.random() < 0.4,
    #     lambda: random.random() < 0.6,
    # ])
    # test.make_iters(int(1e5))

    # test = Thompson([
    #     lambda: random.random() < 0.4,
    #     lambda: random.random() < 0.6,
    # ])
    # test.make_iters(int(1e5))
    
    # actual_std = 1
    # test = GaussianThompson([
    #     lambda: random.normalvariate(0.4, actual_std),
    #     lambda: random.normalvariate(0.6, actual_std),
    # ], actual_std=actual_std)
    # test.make_iters(int(1e5))
    
    # actual_std = 1
    # test = GaussianGreedy([
    #     lambda: random.normalvariate(0.4, actual_std),
    #     lambda: random.normalvariate(0.6, actual_std),
    # ], actual_std=actual_std)
    # test.make_iters(int(1e5))

    print(test.get_stats())
    print("Total time spent:", time.time() - start)
