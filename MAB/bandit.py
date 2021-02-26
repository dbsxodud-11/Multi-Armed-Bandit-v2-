import numpy as np

class MultiArmedBandit :
    def __init__(self, n_arms) :
        self.n_arms = n_arms

    def pull(self, k) : # Get Reward from kth Bandit
        reward = np.zeros(self.n_arms)
        return self.reward[k]

class GaussianMultiArmedBandit(MultiArmedBandit) :
    def __init__(self, n_arms, mu=None, sigma=None) :
        MultiArmedBandit.__init__(self, n_arms)
        if not mu :
            mu = np.random.permutation(np.arange(n_arms))
            sigma = np.abs(np.random.normal(size=n_arms))
        self.mu = mu
        self.sigma = sigma
        print(f"Gaussian Multi-Armed Bandit Initialized")
        for i in range(n_arms) :
            print(f"Arm_{i} - mu: {mu[i]}   sigma: {sigma[i]}")
        print()

    def pull(self, k) :
        return max(np.random.normal(self.mu[k], self.sigma[k]), 0.0)

class BinomialMultiArmedBandit(MultiArmedBandit) :
    def __init__(self, n_arms, n=1, p=None) :
        MultiArmedBandit.__init__(self, n_arms)
        self.n = n
        if not p :
            p = np.random.uniform(size=n_arms)
        self.p = p
    
    def pull(self, k) :
        return np.random.binomial(self.n, p=self.p[k])

class BernoulliMultiArmedBandit(BinomialMultiArmedBandit) :
    def __init__(self, n_arms, p=None) :
        BinomialMultiArmedBandit.__init__(self, n_arms, n=1, p=p)
    
