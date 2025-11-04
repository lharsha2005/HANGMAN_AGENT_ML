import random
import string

class HangmanEnv:
    def __init__(self, word_list, max_wrong_guesses=6):
        self.word_list = word_list
        self.max_wrong_guesses = max_wrong_guesses
        self.alphabet = string.ascii_uppercase
        self.reset()

    def reset(self):
        self.target_word = random.choice(self.word_list)
        self.guessed_letters = set()
        self.wrong_guesses = 0
        self.game_over = False
        self.won = False
        return self._get_state()

    def _get_state(self):
        masked = ''.join([c if c in self.guessed_letters else '_' for c in self.target_word])
        return {
            'masked_word': masked,
            'guessed_letters': self.guessed_letters.copy(),
            'wrong_guesses': self.wrong_guesses,
            'remaining_guesses': self.max_wrong_guesses - self.wrong_guesses,
            'game_over': self.game_over,
            'won': self.won,
            'word_length': len(self.target_word)
        }

    def step(self, action):
        if self.game_over:
            return self._get_state(), 0, True, {}

        action = action.upper()
        if action in self.guessed_letters:
            return self._get_state(), -2, self.game_over, {'repeated': True}

        self.guessed_letters.add(action)

        if action in self.target_word:
            reward = 10
            if all(c in self.guessed_letters for c in self.target_word):
                self.won = True
                self.game_over = True
                reward = 100
        else:
            self.wrong_guesses += 1
            reward = -10
            if self.wrong_guesses >= self.max_wrong_guesses:
                self.game_over = True
                reward = -100

        return self._get_state(), reward, self.game_over, {}
