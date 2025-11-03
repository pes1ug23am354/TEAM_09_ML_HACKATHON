# Quick Start Guide - Hangman ML Hackathon

## 🎯 Goal
Build an intelligent Hangman agent that:
1. Uses HMM to predict letter probabilities
2. Uses RL to make optimal guesses
3. Maximizes win rate while minimizing wrong/repeated guesses

## ⚡ Quick Start (Priority Order)

### Step 1: Environment Setup (2-3 hours)
```python
# Create basic Hangman game
class HangmanEnv:
    - init(word, max_lives=6)
    - get_state() → masked_word, guessed_letters, lives_left
    - guess_letter(letter) → (reward, new_state, done, info)
    - reset()
```

### Step 2: HMM Implementation (4-6 hours)
```python
# Option 1: Use hmmlearn library
from hmmlearn import hmm

# Option 2: Simple character-based model
# Train on corpus.txt to learn:
# - Letter probabilities by position
# - Bigram/trigram probabilities
# - Context-aware letter probabilities
```

**Quick Approach:**
- Train separate simple models for each word length (4-15)
- For each position, learn letter probabilities
- Use smoothed frequencies: P(letter | position, length)

### Step 3: RL Agent - Start Simple (4-6 hours)
```python
# Start with Q-Learning (table-based)
class QLearningAgent:
    - Q-table: dict[state → dict[action → Q-value]]
    - ε-greedy exploration
    - Update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

**State Representation (Simple):**
- `"_PPLE:[ESR]:5"` → string
- Or: tuple(masked_word, tuple(sorted(guessed_letters)), lives)

### Step 4: Integration (2-3 hours)
```python
# Combine HMM + RL
def select_letter(state, hmm_probs, agent):
    # HMM gives probability distribution
    # RL agent uses state + HMM probs to choose action
    action = agent.select_action(state, hmm_probs)
    return action
```

### Step 5: Training Loop (2-4 hours)
```python
for episode in range(num_episodes):
    word = sample_word(corpus)
    env = HangmanEnv(word)
    state = env.get_state()
    
    while not done:
        hmm_probs = hmm.get_probabilities(state)
        action = agent.select_action(state, hmm_probs)
        reward, new_state, done, info = env.guess_letter(action)
        agent.update(state, action, reward, new_state)
        state = new_state
```

### Step 6: Evaluation (1-2 hours)
```python
# Test on test.txt (2000 words)
results = []
for word in test_words:
    score = play_game(word, agent, hmm)
    results.append(score)

# Calculate:
# - Success Rate
# - Total Wrong Guesses
# - Total Repeated Guesses
# - Final Score
```

## 📊 Minimum Viable Solution (MVS)

To get a working solution quickly:

### HMM (Simplified)
- For each word length L (4-20), count letter frequencies at each position
- P(letter | position, length) = count / total_words_of_length

### RL Agent (Simplified)
- Simple Q-learning with ε-greedy
- State: masked_word string + guessed_letters set + lives
- Reward: +1 correct, -1 wrong, +10 win, -5 lose
- Q-table with string-based states

### Training
- Train HMM on corpus.txt (few minutes)
- Train RL agent on subset of corpus (100-1000 episodes)
- Evaluate on test.txt

## 🚀 Week Timeline

### Day 1-2: Basic Implementation
- [x] Hangman environment
- [x] Simple HMM (position-based letter frequencies)
- [x] Simple Q-learning agent
- [x] Basic integration

### Day 3-4: Improvement
- [ ] Better HMM (bigrams/trigrams)
- [ ] Improved RL state representation
- [ ] Tune hyperparameters
- [ ] Train on full corpus

### Day 5-6: Optimization
- [ ] Try DQN (if needed)
- [ ] Feature engineering
- [ ] Reward function tuning
- [ ] Exploration strategy refinement

### Day 7: Finalization
- [ ] Final evaluation on test.txt
- [ ] Create visualizations
- [ ] Write analysis report
- [ ] Prepare for demo/viva

## 🛠️ Essential Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install hmmlearn  # For HMM
# Optional:
pip install torch  # For DQN
pip install jupyter  # For notebooks
```

## 📝 Code Structure

```python
# environment.py
class HangmanEnv:
    """Hangman game environment"""
    pass

# hmm_model.py
class HangmanHMM:
    """HMM for letter probability estimation"""
    def train(corpus):
        pass
    
    def get_probabilities(state):
        """Return P(letter) for each letter given current state"""
        pass

# rl_agent.py
class QLearningAgent:
    """RL agent for Hangman"""
    def select_action(state, hmm_probs):
        pass
    
    def update(state, action, reward, next_state):
        pass

# main.py
def train_agent(corpus, num_episodes=1000):
    hmm = HangmanHMM()
    hmm.train(corpus)
    
    agent = QLearningAgent()
    env = HangmanEnv()
    
    for episode in range(num_episodes):
        # Training loop
        pass
    
    return agent

def evaluate(agent, test_words):
    # Evaluation on test set
    pass
```

## 💡 Key Insights

1. **HMM Purpose**: Provides letter probabilities given current state
   - Example: Given `"_PPLE"`, HMM says P('A')=0.15, P('R')=0.3, etc.

2. **RL Purpose**: Uses HMM probabilities + game state to choose best letter
   - Considers: Which letter maximizes long-term reward?
   - Balances: Exploitation (high prob letters) vs Exploration

3. **Reward Design**: Critical for success
   - Too high on wrong guess: Agent too cautious
   - Too low on wrong guess: Agent too reckless
   - Win bonus should encourage completing games

4. **State Representation**: Balance simplicity vs information
   - Too simple: Agent can't learn well
   - Too complex: Training too slow, overfitting

## 🎓 Learning Resources

- HMM: Hidden Markov Models fundamentals
- RL: Q-Learning, DQN, exploration-exploitation
- Hangman: Classic game mechanics

## ⚠️ Common Pitfalls

1. **Don't overcomplicate HMM initially** - Start simple
2. **Don't forget to handle different word lengths** - This is crucial
3. **Don't ignore repeated guesses** - Important for scoring
4. **Don't train too long without evaluation** - Check test performance early
5. **Don't forget to save your models** - For demo/viva

## ✅ Success Checklist

- [ ] Hangman environment working
- [ ] HMM trained on corpus.txt
- [ ] RL agent training successfully
- [ ] Agent wins > 50% of games on validation set
- [ ] Evaluation on test.txt completed
- [ ] Final score calculated
- [ ] Notebooks with code + results
- [ ] Analysis report written
- [ ] Plots and visualizations ready

Good luck! Start simple, iterate, and improve! 🎯
