import string

def load_words(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        words = [line.strip().upper() for line in f if line.strip()]
    clean = [w for w in words if w.isalpha()]
    print(f"Loaded {len(words)} words, {len(clean)} alphabetic")
    return clean

def preprocess_words(words, min_length=3, max_length=15):
    filtered = [w for w in words if min_length <= len(w) <= max_length and w.isalpha()]
    print(f"Filtered to {len(filtered)} words (length {min_length}-{max_length})")
    return filtered

ALPHABET = string.ascii_uppercase
