import numpy as np

class Policy :
    def __init__(self) :
        print("Base Policy")
    
    def get_action(self, arms) :
        pass

class EpsGreedyPolicy(Policy) :
    def __init__(self, eps) :
        self.eps = eps
    
    def get_action(self, Qs) :
        # e-greedy method
        if np.random.random() < self.eps :
            action = np.random.choice(len(Qs))
        else :
            action = np.argmax(Qs)
        return action
    