import string
import pickle
from collections import defaultdict

def default_action_dict():
    return defaultdict(float)

class QLearningAgent:
    """
    Simple agent that uses HMM probabilities directly (deterministic policy).
    Q-learning hooks kept for project completeness.
    """
    def __init__(self, hmm_model, learning_rate=0.1, discount_factor=0.9):
        self.hmm = hmm_model
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.alphabet = string.ascii_uppercase
        self.q_table = defaultdict(default_action_dict)
        self.common = ['E','T','A','O','I','N','S','H','R','D','L']

    def choose_action(self, state, hmm_probs):
        valid = [l for l in self.alphabet if l not in state['guessed_letters']]
        if not valid:
            return None
        if hmm_probs and sum(hmm_probs.values()) > 0:
            valid_probs = {k: v for k, v in hmm_probs.items() if k in valid}
            if valid_probs:
                return max(valid_probs, key=valid_probs.get)
        for l in self.common:
            if l in valid:
                return l
        return valid[0]

    def learn(self, state, hmm_probs, action, reward, next_state, next_hmm_probs, done):
        pass

    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_q_table(self, filename):
        try:
            with open(filename, 'rb') as f:
                self.q_table = defaultdict(default_action_dict, pickle.load(f))
        except FileNotFoundError:
            pass
