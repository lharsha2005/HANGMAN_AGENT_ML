import string
from collections import Counter

class HangmanHMM:
    """
    Pattern-matching HMM variant for Hangman.
    - Indexes corpus by word length
    - Exact pattern match -> letter frequencies
    - Partial match fallback
    - Smart fallback (vowels early, common consonants late, global frequency + positional prior)
    """
    def __init__(self):
        self.corpus_words = []
        self.words_by_length = {}
        self.alphabet = set(string.ascii_uppercase)
        self.letter_freq = {
            'E': 12.7, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97, 'N': 6.75,
            'S': 6.33, 'H': 6.09, 'R': 5.99, 'D': 4.25, 'L': 4.03, 'C': 2.78,
            'U': 2.76, 'M': 2.41, 'W': 2.36, 'F': 2.23, 'G': 2.02, 'Y': 1.97,
            'P': 1.93, 'B': 1.49, 'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15,
            'Q': 0.10, 'Z': 0.07
        }

    def train(self, words):
        print("\n=== PATTERN MATCHING HMM ===")
        self.corpus_words = [w for w in words if w.isalpha()]
        self.words_by_length.clear()
        for w in self.corpus_words:
            self.words_by_length.setdefault(len(w), []).append(w)
        lengths = sorted(self.words_by_length.keys())
        print(f"Corpus: {len(self.corpus_words)} words | Lengths: {lengths[0]}..{lengths[-1]}")
        print("=== READY ===\n")

    def predict_probabilities(self, masked_word, guessed_letters, word_length=None):
        # 1) Exact pattern match
        matches = self._find_matches(masked_word, guessed_letters)
        # 2) Partial match if few/no exact
        if len(matches) < 5:
            matches = self._partial_matches(masked_word, guessed_letters)
        # 3) Smart fallback if still none
        if not matches:
            return self._smart_fallback(masked_word, guessed_letters)

        counts = Counter()
        for word in matches:
            for ch in word:
                if ch not in guessed_letters:
                    counts[ch] += 1
        total = sum(counts.values())
        if total == 0:
            return self._smart_fallback(masked_word, guessed_letters)
        return {ch: c / total for ch, c in counts.items()}

    def _find_matches(self, pattern, guessed):
        L = len(pattern)
        if L not in self.words_by_length:
            return []
        res = []
        for w in self.words_by_length[L]:
            match = True
            for wc, pc in zip(w, pattern):
                if pc == '_':
                    if wc in guessed:
                        match = False
                        break
                elif wc != pc:
                    match = False
                    break
            if match:
                res.append(w)
        return res

    def _partial_matches(self, pattern, guessed):
        L = len(pattern)
        if L not in self.words_by_length:
            return []
        res = []
        for w in self.words_by_length[L]:
            ok = True
            for wc, pc in zip(w, pattern):
                if pc != '_' and wc != pc:
                    ok = False
                    break
            if ok:
                res.append(w)
        return res[:50]

    def _smart_fallback(self, masked_word, guessed):
        blanks = masked_word.count('_')
        L = len(masked_word) if masked_word else 0
        ratio = (blanks / L) if L else 1.0

        # Positional prior: mid-position letters slightly favored
        mid_prior = {
            'A': 0.02, 'E': 0.03, 'I': 0.02, 'O': 0.02,
            'N': 0.02, 'R': 0.02, 'S': 0.02, 'T': 0.02, 'L': 0.02
        }

        # A) Early game -> stronger vowels
        if ratio > 0.6:
            vowels = {'E': 0.33, 'A': 0.27, 'I': 0.16, 'O': 0.16, 'U': 0.08}
            avail = {l: p for l, p in vowels.items() if l not in guessed}
            if avail:
                for l in list(avail.keys()):
                    avail[l] += mid_prior.get(l, 0.0)
                s = sum(avail.values())
                return {l: p / s for l, p in avail.items()}

        # B) Mid game -> balanced mix
        if 0.3 <= ratio <= 0.6:
            mix = {
                'E': 0.12, 'A': 0.10, 'I': 0.08, 'O': 0.08, 'U': 0.05,
                'T': 0.10, 'R': 0.10, 'S': 0.09, 'N': 0.09, 'L': 0.07,
                'D': 0.07, 'H': 0.05
            }
            avail = {l: p for l, p in mix.items() if l not in guessed}
            if avail:
                for l in list(avail.keys()):
                    avail[l] += mid_prior.get(l, 0.0)
                s = sum(avail.values())
                return {l: p / s for l, p in avail.items()}

        # C) Late game -> strong consonant focus
        if ratio < 0.3:
            cons = {'T': 0.17, 'R': 0.15, 'S': 0.13, 'N': 0.13, 'L': 0.11, 'D': 0.10, 'H': 0.08, 'C': 0.07, 'M': 0.06}
            avail = {l: p for l, p in cons.items() if l not in guessed}
            if avail:
                s = sum(avail.values())
                return {l: p / s for l, p in avail.items()}

        # D) Global frequency + positional prior
        avail = {l: f for l, f in self.letter_freq.items() if l not in guessed}
        if avail:
            for l in list(avail.keys()):
                avail[l] += mid_prior.get(l, 0.0)
            s = sum(avail.values())
            return {l: f / s for l, f in avail.items()}

        # E) Uniform fallback
        remaining = [l for l in self.alphabet if l not in guessed]
        return {l: 1.0 / len(remaining) for l in remaining} if remaining else {}
