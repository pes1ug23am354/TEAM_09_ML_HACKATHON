## Complete_Solution.ipynb — Cell-by-Cell Explanation (Detailed)

This document explains the structure and logic of the `notebooks/Complete_Solution.ipynb` notebook in detail. It covers data loading, preprocessing, the Hangman environment, the HMM model, the hybrid Q-learning + HMM agent, training, and evaluation.

Note: Cell indices in Jupyter can shift as edits are made. Use the section headers and key phrases inside each cell to locate them.

---

### Cell: Title and Execution Order Warnings
- Purpose: Introduces the notebook and emphasizes running cells top-to-bottom.
- Key content:
  - Explains common error: running a cell that uses `FirstOrderHMM` before its definition.
  - Execution order checklist.

Why important: Ensures the notebook’s dependencies (classes/functions) are defined before use.

---

### Cell: Project Overview (Markdown)
- Purpose: High-level overview of the solution architecture.
- Key content:
  - Hidden Markov Model (HMM) for dynamic letter probabilities.
  - Reinforcement Learning (Q-Learning) agent for decision making.
  - Hybrid system: combines HMM + Q for robust guessing.
  - Overfitting prevention and online learning.

---

### Cell: Imports and Globals
- Purpose: Import Python libraries and set deterministic seeds.
- Libraries: numpy, pandas (optional), matplotlib, seaborn, collections, random, pickle, os, sys.
- Settings: `%matplotlib inline`, styling, `np.random.seed(42)`, `random.seed(42)`.

Why important: Reproducibility and consistent plotting.

---

### Cell: 1. Load Data
- Purpose: Locate and load `Data/corpus.txt` and `Data/test.txt` robustly.
- Logic:
  - Tries multiple absolute/relative paths to find the files.
  - Prints resolved paths and counts.
  - Loads `corpus_raw` and `test_words_raw` as lists of non-empty lines.
  - Guards to raise a clear error if either is empty.

Why important: Prevents silent failures when running from different working directories (local vs Colab).

---

### Cell: 1.0 Data Cleaning and Preprocessing (Functions)
- Purpose: Define preprocessing functions used on both corpus and test sets.
- Functions:
  - `normalize_case(word, case_mode='lower')`: case normalization.
  - `remove_non_alphabetic` (integrated logic): drops non a–z characters.
  - `fix_common_typos` (optional, simple heuristics): can map common patterns.
  - `is_valid_word(word, min_length=3, max_length=20)`: length filtering.
  - `remove_duplicates_keep_order(words)`: stable de-duplication.
  - `clean_and_preprocess_word(word, ...)`: pipelines the above.
  - `preprocess_word_list(words, ...)`: applies cleaning to a list and returns `(cleaned_words, stats)`.
- Robustness fix: Handles empty input (returns empty stats) and prints “Removal rate: N/A” when original count is zero.

Why important: Ensures training/evaluation receive clean, comparable tokens.

---

### Cell: Preprocess Corpus
- Purpose: Apply preprocessing to `corpus_raw`.
- Outputs:
  - `corpus`: cleaned list.
  - `corpus_stats`: counts, duplicates removed, removed samples (optional print).

---

### Cell: Preprocess Test Set
- Purpose: Same preprocessing for `test_words_raw` → `test_words`.
- Extra check: Leakage checks (optional) and monotone logging.

---

### Cell: 1.1 Data Quality Analysis (Optional)
- Purpose: Summary stats (length histograms, counts) and sanity checks.
- Helps detect skew or anomalies.

---

### Cell: 1.2 Train/Validation Split
- Purpose: 90/10 split from preprocessed `corpus` → `training_corpus`, `validation_corpus`.
- Robustness guards:
  - Ensures non-empty train/val; if needed, reconstructs from `corpus`.
  - Ensures `test_words` non-empty (recovers from `test_words_raw` if available).

---

### Cell: 1.3 Bucketing by Word Length
- Purpose: Produce buckets `{length → [words]}` for `training_corpus`, `validation_corpus`, `test_words`.
- Why: Helpful for sampling diverse lengths and for candidate filtering later.

---

### Cell: 2. Hangman Environment (`HangmanEnv`)
- Purpose: Minimal environment simulation.
- Methods:
  - `get_state()`: returns dict with `masked_word`, `guessed_letters`, `lives_left`, `word_length`, `num_guesses`.
  - `guess_letter(letter)`: updates state, returns `(reward, new_state, done, info)`.
  - `reset(word=None)`: reinitializes for a new word.
- Reward design: Kept as requested (unchanged). Positive for correct, negative for wrong, bonus on win.

---

### Cell: 3. HMM Model (`FirstOrderHMM`)
- Purpose: First-order character HMM providing dynamic letter probabilities.
- Training (`train`):
  - Counts bigrams, starts/ends, positional frequencies.
  - Computes smoothed probabilities.
- Inference:
  - `get_letter_probability_given_prev`, `..._given_next`, `..._by_position`.
  - `get_probabilities_for_mask(masked_word, guessed_letters)`: core method combining:
    - Bidirectional context (prev and next known letters).
    - Strong bigram boosts (e.g., TH/HE/CH/SH/WH, AP/AL/AN/AR) with aggressive multipliers.
    - Position-based priors.
    - Top-prediction sharpening (boost for clear winner).
- Evaluation helpers: `calculate_log_likelihood`, `calculate_perplexity`.

Why important: Produces context-aware, per-word dynamic probabilities (no fixed global letter frequencies).

---

### Cell: 3.1 HMM Training + Perplexity
- Purpose: Train HMM on `training_corpus`; compute training/validation perplexity to check for over/under-fitting.

---

### Cell: Candidate-Based Probability Module (Hybrid Combiner)
- Purpose: Narrow predictions to only words that fit the current mask and guessed letters, then mix that distribution with HMM.
- Components:
  - `CANDIDATE_BUCKETS = bucket_words_by_length(training_corpus)` for fast lookup.
  - `filter_candidates(masked_word, guessed_letters, buckets)`: regex-based filter using known/unknown positions, excludes letters known to be wrong.
  - `letter_distribution_from_candidates(...)`: frequency over blank positions from remaining candidates; includes an information-gain boost (letters present in ~50% of candidates are preferred as they split the search best).
  - `combine_hmm_and_candidates(hmm, masked_word, guessed_letters, buckets, mix_weight)`: convex mix with weight `CANDIDATE_MIX_WEIGHT` (default ~0.70 for candidates, 0.30 for HMM).
  - `get_letter_probs(hmm, state)`: unified function used by training/evaluation to fetch combined probabilities.

Why important: Greatly improves accuracy by leveraging the filtered candidate set, while staying robust through HMM blending.

---

### Cell: Candidate Lock-In Helper
- Purpose: If only one candidate remains, pick the next missing letter from that word.
- Function:
  - `next_letter_from_single_candidate(state)` returns the next uncovered position’s character when exactly one candidate matches the mask.

Why important: Eliminates wasted guesses once the word is identified.

---

### Cell: 4. RL Agent Configuration (`RL_CONFIG`)
- Fields:
  - `learning_rate` (α), `discount_factor` (γ), `epsilon`, `epsilon_decay`, `epsilon_min`.
  - `hmm_weight`, `q_weight`: weights for hybrid action scoring.
- Tuning notes:
  - Higher `hmm_weight` biases towards more reliable HMM guidance.
  - `q_weight` scales learned Q-values.

---

### Cell: EVAL_MODE
- Purpose: Choose evaluation mode.
- Values:
  - `'hybrid'`: uses `q_weight*Q + hmm_weight*HMM_prob` (default).
  - `'hmm'`: forces pure HMM-greedy (useful for comparison).

---

### Cell: 4. Q-Learning Agent (`QLearningAgent`)
- Purpose: Table-based Q-learning agent with online updates.
- State key: `masked_word:word_len:sorted(guessed):lives_left`.
- Action selection (`select_action`):
  1. Candidate lock-in: if exactly one candidate remains, return its next missing letter.
  2. Low-lives safeguard: if `lives_left <= 2`, pick HMM-greedy to avoid wrong guesses.
  3. Exploration (ε): information-gain weighted exploration guided by HMM (prefers letters that split candidates near 50/50; uses top-K sampling).
  4. Exploitation: hybrid scoring = `combined = q_weight*Q(s,a) + hmm_weight*P_HMM(a|state)`; pick argmax.
- Update: standard Q-learning update, done after each guess (online learning).

Why important: Marries learned long-term behavior with reliable HMM guidance.

---

### Cell: 5. RL Training Setup
- Purpose: Prepare `rl_training_words` from buckets with guards to avoid empty lists.
- Also samples `val_subset` for periodic validation checks, with fallback if needed.
- Prints configuration: number of training words, episodes, validation set size.

---

### Cell: 5.1 Training Loop — Online Learning
- Purpose: Main reinforcement learning loop across episodes.
- Flow per episode:
  - Sample `word` (safe fallback to `training_corpus` if `rl_training_words` empty).
  - Create `HangmanEnv(word)`; get initial `state`.
  - Per step:
    - Compute combined probabilities: `hmm_probs = get_letter_probs(hmm, state)`.
    - `action = agent.select_action(state, hmm_probs)` using hybrid strategy.
    - Step env: `(reward, next_state, done, info) = env.guess_letter(action)`.
    - Recompute probs for new state: `next_hmm_probs = get_letter_probs(hmm, next_state)`.
    - Update Q-values immediately: `agent.update(...)`.
  - Track per-episode reward, wrong guesses, steps.
  - Periodically evaluate on `val_subset` with greedy policy (no exploration) to monitor generalization.

Why important: Ensures learning occurs after each guess and HMM guidance is updated with newly revealed letters.

---

### Cell: 6. Training Visualizations (Optional)
- Purpose: Plots of reward curves, win rates, epsilon decay, wrong guesses trends.
- Diagnostic messages: overfitting check, training health report.

---

### Cell: 7. Evaluation on Test Set
- Purpose: Evaluate the trained agent on `test_words`.
- Guards:
  - If `test_words` empty, attempts to recover from `test_words_raw`, else uses subsets from val/train.
  - Divisions by `len(test_words)` are guarded to avoid ZeroDivisionError.
- Calls: `evaluation_results = evaluate_agent(agent, hmm, test_words, max_lives=6)`.
- Metrics:
  - Success rate, total wrong guesses, repeated guesses, final score per formula.

Tips:
- Compare `EVAL_MODE = 'hybrid'` vs `'hmm'` to see the value of Q-learning.
- Positive final score depends on both higher success rate and fewer wrong guesses.

---

### Cell: 8. Save Models (Optional)
- Purpose: Save HMM parameters and Q-table to `models/`.
- Useful for reusing trained agents without retraining.

---

### Cell: 9. Additional Analysis (Optional)
- Purpose: Performance by length buckets, wrong guesses distribution, correlations.
- Helps pinpoint where to tune further (e.g., very long or very short words).

---

## Design Highlights and Rationale

- Dynamic probabilities per word: No global fixed letter frequencies; everything depends on the active mask, guessed letters, and word length.
- Candidate filtering: Big win in accuracy; combined with HMM to avoid overfitting on small candidate sets.
- Information-gain exploration: Chooses letters that maximally split remaining candidates to gather information faster.
- Safety heuristics: Low-lives HMM-greedy and lock-in on single candidate reduce wrong guesses, boosting final score without changing rewards.
- Hybrid action scoring: `argmax(q_weight*Q + hmm_weight*HMM)`. Q captures long-term effects (lives, mask evolution), HMM captures immediate letter plausibility.

## Parameter Tuning Guide

- `RL_CONFIG['hmm_weight']` (suggested 30–45): Increase to lean more on HMM when Q is uncertain.
- `RL_CONFIG['q_weight']` (0.8–1.2): Adjust finer influence of learned Q-values.
- `CANDIDATE_MIX_WEIGHT` (0.65–0.75): More candidate weight improves specificity; too high may overfit when few candidates remain.
- `epsilon_decay`: 0.9995–0.9998 provides longer exploration when needed.
- Episodes: more episodes stabilize Q-values; ensure validation performance keeps improving.

## Troubleshooting

- Empty datasets: The notebook contains guards to reconstruct train/val/test from available sources.
- NameError for HMM: Run cells in order; ensure HMM class cell ran before training/evaluation cells.
- Negative final score: Use `'hybrid'` evaluation, verify candidate module is active, increase `hmm_weight`, and ensure low-lives safeguard is in place.

## How to Compare Modes Quickly

1. Set `EVAL_MODE = 'hmm'` → run evaluation; note success rate and wrong guesses.
2. Set `EVAL_MODE = 'hybrid'` → run evaluation; compare improvements due to Q-learning.

---

This document should let you navigate the notebook confidently, understand each component’s role, and know exactly where to tune for better success rate and positive final score.
