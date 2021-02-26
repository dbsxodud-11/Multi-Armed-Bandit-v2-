import numpy as np

class Environment :
    def __init__(self, bandit, agents) :
        self.bandit = bandit
        self.agents = agents

    def reset(self) :
        for agent in self.agents :
            agent.reset()

    def run(self, trials, experiments) :
        scores = np.zeros((len(self.agents), trials))

        for _ in range(experiments) :
            self.reset()
            for t in range(trials) :
                for i, agent in enumerate(self.agents) :
                    action = agent.get_action()
                    reward = self.bandit.pull(action)
                    agent.train(reward)

                    scores[i, t] += reward
        return scores/experiments