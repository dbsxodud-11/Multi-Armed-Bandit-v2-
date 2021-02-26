import numpy as np

class Agent:
    def __init__(self, bandit, policy):
        self.bandit = bandit
        self.policy = policy

        self.Qs = np.zeros(self.bandit.n_arms)
        self.Ns = np.zeros(self.bandit.n_arms)
        self.t = 0

        self.last_action = None

    def reset(self) :
        self.Qs = np.zeros(self.bandit.n_arms)
        self.Ns = np.zeros(self.bandit.n_arms)

    def get_action(self) :
        action = self.policy.get_action(self)
        self.last_action = action
        return action
    
    def train(self, reward) :
        # Update Value Function
        self.Ns[self.last_action] += 1
        self.Qs[self.last_action] += (1/self.Ns[self.last_action]) * (reward - self.Qs[self.last_action])
        self.t += 1


class TSAgent(Agent) :
    def __init__(self, bandit, policy) :
        Agent.__init__(self, bandit, policy) 
        self.alpha = np.ones(self.bandit.n_arms)
        self.beta = np.ones(self.bandit.n_arms)
        self.Qs = np.random.beta(self.alpha, self.beta)

    def reset(self) :
        self.Qs = np.zeros(self.bandit.n_arms)
        self.Ns = np.zeros(self.bandit.n_arms)
        self.alpha = np.ones(self.bandit.n_arms)
        self.beta = np.ones(self.bandit.n_arms)

    def train(self, reward) :
        # Update Posterior Distribution
        self.Ns[self.last_action] += 1
        self.alpha[self.last_action] += reward
        self.beta[self.last_action] -= reward-1
        self.Qs = np.random.beta(self.alpha, self.beta)
        self.t += 1