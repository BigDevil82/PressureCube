class EpsilonScheduler:
    def __init__(self):
        self.epsilon = 0.5
        self.decay_rate = 0.99
        self.min_epsilon = 0.01

    def decay(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

    def get_epsilon(self):
        self.decay()
        return self.epsilon
