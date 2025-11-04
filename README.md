# 🧠 Hackman – Hangman Reinforcement Learning Agent

**Course:** UE23CS352A — Machine Learning Hackathon  
**Project Title:** Hackman (Hangman RL Agent)  
**Team Members:**  
- KP Vishnu (PES2UG23CS295)  
- KVN Deepak (PES2UG23CS293)  
- L. Harshavardhan (PES2UG23CS303)  
- L. Teja (PES2UG23CS297)  
**Section:** E  
**Date:** 03-11-2025  

---

## 📘 Overview

**Hackman** is an intelligent Hangman agent that guesses hidden English words efficiently under a limited error budget.  
It combines:
- A **pattern-matching HMM (Hidden Markov Model)** to compute letter probabilities, and  
- A **lightweight RL (Reinforcement Learning) agent** that greedily selects the next best letter.  

The system emphasizes **speed, modularity, and interpretability**, achieving a **14–20% success rate** on a challenging out-of-distribution (OOD) word set.

---

## 🎯 Objective

- Build an AI agent that plays **Hangman optimally** — maximize solved words, minimize wrong guesses.  
- Operate **offline** (no internet or API calls).  
- Generate metrics: success rate, wrong guesses, and composite score.  
- Provide **explainable outputs** with top-k probability traces.

---

## 🧩 System Components

### 1. HMM Pattern-Matching Scorer
- Filters a corpus for words matching the current masked pattern.  
- Estimates probabilities for unguessed letters.  
- Uses fallbacks:
  - Early: vowel-heavy priors  
  - Mid-game: balanced vowels/consonants  
  - Late-game: consonant-heavy  
- Optional bigram and positional priors.  
- Deterministic, interpretable, and fast.

### 2. RL Agent
- **State:** masked word, guessed set, wrong-guess count, word length.  
- **Action:** choose an unguessed letter.  
- **Policy:** greedy selection on HMM probabilities with deterministic tie-breaks.  
- **Reward shaping:**  
  ```
  +10 correct guess
  -10 wrong guess
  -2 repeated guess
  +100 win
  -100 loss
  ```

### 3. Environment
- Simulates Hangman rounds.  
- Tracks guesses, lives, and rewards.  
- Configurable maximum wrong guesses (default: 6).

---

## 🧮 Dataset & Preprocessing

| Split | Count | Notes |
|--------|--------|--------|
| Training Corpus | ~48,446 words | 3–15 letters, alphabetic only |
| Test Set | 2,000 words | 0% overlap, OOD words like “GASTROSTENOSIS” |

**Steps:**
- Uppercase normalization  
- Non-alphabetic token removal  
- Word length filtering  
- Indexing by word length for O(1) lookup  

---

## 🏗️ Project Structure

```
Hackman/
│
├── src/
│   ├── utils.py              # Data loading and preprocessing
│   ├── hmm_model.py          # HMM-based pattern matcher and scorer
│   ├── hangman_env.py        # Game simulation environment
│   └── rl_agent.py           # RL agent for action selection
│
├── notebooks/
│   ├── 1_Train.ipynb         # Build HMM index and test
│   ├── 2_Evaluate.ipynb      # Evaluate model on test set
│   └── 3_Demo.ipynb          # Step-wise interactive demo
│
├── models/
│   └── hmm_model.pkl         # Saved model (generated)
│
├── results/
│   ├── final_score.txt       # Success rate, avg wrong, composite score
│   └── demo_trace.txt        # Step-wise log of predictions
│
└── README.md                 # Project documentation
```

---

## ⚙️ How to Run

### 🧰 Requirements
- Python 3.x  
- Libraries: `numpy`, `matplotlib` (optional for visualization)

### 🚀 Steps

```bash
# 1️⃣ Build model index
jupyter notebook notebooks/1_Train.ipynb

# 2️⃣ Evaluate performance
jupyter notebook notebooks/2_Evaluate.ipynb

# 3️⃣ Run interactive demo
jupyter notebook notebooks/3_Demo.ipynb
```

Outputs will be saved in the `results/` folder:
- `final_score.txt` — aggregated evaluation metrics  
- `demo_trace.txt` — top-k probabilities and decisions per step  

---

## 📊 Key Results

| Metric | Value |
|--------|--------|
| Success Rate | ~14.1% |
| Avg Wrong Guesses | ~5.7 per game |
| Composite Score | Negative (due to OOD difficulty) |

---

## 🔍 Error Analysis

- **Failure Cases:** rare word stems (e.g., OB-, PYR-) lead to poor prior predictions.  
- **Diagnostics:** top-k probability logs, wrong-guess trajectories, and candidate set tracking.

---

## 🚧 Future Improvements

- Expand corpus with **domain-specific lexicons** (medical, legal, scientific).  
- Integrate **character-level language models** for semantic generalization.  
- Add **prefix/suffix detectors** and **syllable features**.  
- Replace greedy policy with **entropy-based (information gain) selection**.  
- Implement **beam search** for multi-letter lookahead.

---

## 🧩 Reproducibility

- Deterministic runs with fixed seeds.  
- All required scripts and notebooks included.  
- No internet dependency.

---

## 🏁 Conclusion

The Hackman agent demonstrates a **modular, explainable, and fast baseline** for Hangman-playing AI.  
Its HMM+RL framework provides a strong foundation for future semantic or neural enhancements to generalize beyond dictionary-based pattern matching.
