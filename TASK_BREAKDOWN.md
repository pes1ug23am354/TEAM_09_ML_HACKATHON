# Hangman ML Hackathon - Complete Task Breakdown

## Overview
Build an intelligent Hangman agent that combines:
1. **Hidden Markov Model (HMM)** - for letter probability estimation
2. **Reinforcement Learning (RL)** - for optimal letter guessing strategy

## Evaluation Metrics
Your agent will be scored on **2000 test words** with **6 wrong guesses allowed** per game.

**Final Score Formula:**
```
Final Score = (Success Rate × 2000) - (Total Wrong Guesses × 5) - (Total Repeated Guesses × 2)
```

## Step-by-Step Implementation

### Phase 1: Environment Setup (Week 1 - Days 1-2)

#### 1.1 Create Hangman Game Environment
- [ ] Define game state: masked word, guessed letters, lives remaining
- [ ] Implement action space: 26 letters (A-Z), exclude already guessed
- [ ] Create reward function:
  - Positive reward for correct guesses
  - Negative reward for wrong guesses
  - Bonus for winning the game
  - Penalty for repeated guesses
- [ ] Implement game logic: word masking, letter checking, win/loss conditions

**Key Considerations:**
- State representation: `"_PPLE"` (masked word) + guessed letters set + lives left
- Actions: Letters not yet guessed
- Rewards: 
  - +1 for correct guess
  - -1 for wrong guess
  - +10 for winning
  - -0.5 for repeated guess

#### 1.2 Data Loading
- [ ] Load and preprocess `corpus.txt` (50,000 words)
- [ ] Load `test.txt` (2000 words) for evaluation
- [ ] Handle word preprocessing: lowercase, remove special characters

### Phase 2: Hidden Markov Model (Week 1 - Days 2-4)

#### 2.1 Design HMM Structure
Decide on:
- **Hidden States:** What represents the position/context in a word?
  - Option 1: Character positions (1st, 2nd, 3rd letter, etc.)
  - Option 2: Character n-grams (bigrams, trigrams)
  - Option 3: Position relative to word length
  
- **Observations:** The actual letters we observe
- **Emissions:** Probability of observing a letter at each state

#### 2.2 Handle Different Word Lengths
- [ ] Strategy 1: Train separate HMMs for different word lengths (length 4-15+)
- [ ] Strategy 2: Use padding/special tokens to standardize lengths
- [ ] Strategy 3: Normalize positions (0.0 to 1.0 relative to word length)

#### 2.3 Train HMM
- [ ] Implement HMM using `hmmlearn` library or from scratch
- [ ] Train on corpus.txt
- [ ] Learn transition probabilities between states
- [ ] Learn emission probabilities (letter → position)

#### 2.4 HMM Inference
- [ ] Implement function to estimate letter probabilities given:
  - Current masked word pattern (e.g., `"_PPLE"`)
  - Already guessed letters
  - Word length
- [ ] Output: Probability distribution over alphabet (26 letters)

### Phase 3: Reinforcement Learning Agent (Week 1 - Days 4-6)

#### 3.1 Define State Representation
Combine multiple features:
- [ ] Masked word encoding (one-hot or character embeddings)
- [ ] Binary vector of guessed letters (26 dimensions)
- [ ] Lives remaining (scalar)
- [ ] HMM probability distribution (26 dimensions)
- [ ] Word length (scalar)

**Simple Option:** String representation like `"_PPLE:[ESR]:5"`
**Complex Option:** Concatenated feature vector

#### 3.2 Design Reward Function
Balance between:
- [ ] Winning the game (high positive reward)
- [ ] Correct guesses (small positive reward)
- [ ] Wrong guesses (negative reward)
- [ ] Repeated guesses (penalty)

**Example:**
```python
Reward = {
    'correct_guess': +0.5,
    'wrong_guess': -1.0,
    'win': +10.0,
    'lose': -5.0,
    'repeat_guess': -0.5
}
```

#### 3.3 Choose RL Algorithm

**Option A: Q-Learning (Table-based)**
- [ ] Suitable if state space is small/discretizable
- [ ] Implement Q-table: State × Action → Q-value
- [ ] Update rule: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

**Option B: Deep Q-Network (DQN)**
- [ ] Use neural network to approximate Q-function
- [ ] Implement experience replay buffer
- [ ] Implement target network for stability
- [ ] Libraries: PyTorch or TensorFlow

#### 3.4 Exploration Strategy
- [ ] Implement ε-greedy exploration:
  - Start with high ε (0.9) - explore a lot
  - Decay ε over time (exponential decay)
  - Final ε (0.01-0.05) - mostly exploit
- [ ] Consider: UCB (Upper Confidence Bound), Thompson Sampling

#### 3.5 Training Loop
- [ ] For each episode:
  1. Sample a random word from corpus
  2. Initialize game state
  3. While game not over:
     - Get HMM probabilities for current state
     - Agent selects action (letter) based on current state + HMM probs
     - Execute action, get reward, observe new state
     - Update Q-values or neural network
  4. Log metrics (reward, win/loss, wrong guesses)

### Phase 4: Integration (Week 1 - Day 6-7)

#### 4.1 Combine HMM + RL
- [ ] HMM provides probability distribution over letters
- [ ] RL agent uses this distribution + game state to make decision
- [ ] Can weight HMM probabilities in state representation

#### 4.2 Training Schedule
- [ ] Train HMM first (on corpus.txt)
- [ ] Then train RL agent (use corpus words as training games)
- [ ] Iterate: Improve HMM → Retrain RL → Evaluate → Repeat

### Phase 5: Evaluation (Week 1 - Day 7)

#### 5.1 Test on Test Set
- [ ] Run agent on all 2000 words from `test.txt`
- [ ] Track for each game:
  - Win/Loss
  - Number of wrong guesses
  - Number of repeated guesses
  - Total guesses

#### 5.2 Calculate Metrics
- [ ] Success Rate = (Wins / 2000) × 100
- [ ] Total Wrong Guesses (sum across all games)
- [ ] Total Repeated Guesses (sum across all games)
- [ ] Final Score = (Success Rate × 2000) - (Wrong × 5) - (Repeated × 2)

### Phase 6: Deliverables (Week 1 - Day 7)

#### 6.1 Jupyter Notebook
Create comprehensive notebook with:

**Section 1: HMM Implementation**
- [ ] HMM architecture explanation
- [ ] Training code on corpus.txt
- [ ] Visualization of learned probabilities
- [ ] Example: Given `"_PPLE"`, show letter probabilities

**Section 2: RL Agent Implementation**
- [ ] Environment code
- [ ] State/Action/Reward definitions
- [ ] RL algorithm (Q-learning or DQN)
- [ ] Training loop
- [ ] Hyperparameters

**Section 3: Results**
- [ ] Final Score
- [ ] Success Rate
- [ ] Average Wrong Guesses per game
- [ ] Average Repeated Guesses per game
- [ ] Plots:
  - Learning curve (reward per episode)
  - Success rate over training
  - Wrong guesses over time
  - Q-value heatmaps (if table-based)

**Section 4: Analysis**
- [ ] Best/worst performing word types
- [ ] Common failure patterns
- [ ] Agent behavior analysis

#### 6.2 Analysis Report PDF
Write report answering:

**Key Observations:**
- [ ] What were the most challenging parts?
- [ ] What insights did you gain?
- [ ] What patterns did you notice?

**Strategies:**
- [ ] HMM design choices and rationale
- [ ] Why chosen structure (states, emissions)?
- [ ] How different word lengths handled?
- [ ] RL state representation rationale
- [ ] Reward function design and tuning
- [ ] Why chosen RL algorithm?

**Exploration:**
- [ ] Exploration-exploitation strategy
- [ ] How ε-decay schedule chosen?
- [ ] Balance between exploration and exploitation?

**Future Improvements:**
- [ ] What would you do with more time?
- [ ] Advanced techniques to try (PPO, Actor-Critic, etc.)
- [ ] Better HMM structures?
- [ ] Feature engineering improvements?

## Technical Stack Recommendations

### Python Libraries
```python
# HMM
from hmmlearn import hmm  # or implement from scratch

# RL
import numpy as np
import torch  # for DQN
import tensorflow as tf  # alternative

# Utilities
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
```

### Project Structure
```
ml-hackathon/
├── Data/
│   ├── corpus.txt
│   └── test.txt
├── notebooks/
│   ├── 01_HMM_Implementation.ipynb
│   ├── 02_RL_Agent.ipynb
│   ├── 03_Training.ipynb
│   ├── 04_Evaluation.ipynb
│   └── 05_Complete_Solution.ipynb  # Main notebook
├── src/
│   ├── environment.py  # Hangman game
│   ├── hmm_model.py   # HMM implementation
│   ├── rl_agent.py    # RL agent
│   └── utils.py       # Helper functions
├── Analysis_Report.pdf
└── README.md
```

## Quick Start Checklist

- [ ] Day 1: Set up environment, implement Hangman game
- [ ] Day 2-3: Design and train HMM
- [ ] Day 3-4: Implement RL agent skeleton
- [ ] Day 4-5: Integrate HMM + RL, start training
- [ ] Day 5-6: Tune hyperparameters, improve performance
- [ ] Day 6-7: Evaluate on test set, create visualizations
- [ ] Day 7: Write analysis report, prepare for demo/viva

## Success Tips

1. **Start Simple**: Begin with basic Q-learning and simple HMM, then improve
2. **Iterate Fast**: Test on small subset first, then scale up
3. **Visualize**: Plot everything - learning curves, letter frequencies, etc.
4. **Document**: Comment code well, track experiments
5. **Test Early**: Evaluate on test set periodically during development
6. **Balance**: Don't over-optimize one component; balance HMM and RL

## Evaluation Reminder

- 2000 test words
- 6 lives per game
- Score heavily rewards success but penalizes wrong/repeated guesses
- Goal: High success rate + low inefficiency

Good luck! 🎯
