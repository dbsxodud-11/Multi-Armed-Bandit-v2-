import numpy as np

class Policy :
    def __init__(self) :
        print("Base Policy")
    
    def get_action(self, agent) :
        pass

class EpsGreedyPolicy(Policy) :
    def __init__(self, eps) :
        self.eps = eps
    
    def get_action(self, agent):
        # e-greedy method
        Qs = agent.Qs
        if np.random.random() < self.eps :
            action = np.random.choice(len(Qs))
        else :
            action = np.argmax(Qs)
        return action

class GreedyPolicy(EpsGreedyPolicy) :
    def __init__(self) :
        EpsGreedyPolicy.__init__(self, eps=0)

class RandomPolicy(EpsGreedyPolicy) :
    def __init__(self) :
        EpsGreedyPolicy.__init__(self, eps=1)

class UCBPolicy(Policy) :
    def __init__(self, c_level) :
        self.c_level = c_level

    def get_action(self, agent) :
        # UCB method
        Qs = agent.Qs
        Ns = agent.Ns
        exploration = np.log(np.sum(Ns)+1) / Ns
        exploration = self.c_level*np.sqrt(exploration)
        
        new_Qs = Qs + exploration
        action = np.argmax(new_Qs)
        return action