# RL Environment Design Guide - Hangman

## 🎯 Overview

This guide covers potential problems and parameter choices when building the RL environment for Hangman.

## ⚠️ Common Problems & Solutions

### Problem 1: State Space Explosion

**Issue:** 
- State representation too complex → Q-table becomes huge (millions of states)
- Training becomes slow or impossible
- Memory overflow

**Example Bad State:**
```python
# BAD: Too many states
state = {
    'masked_word': '_PPLE',      # Could be any pattern
    'guessed_letters': {'a', 'e', 'p'},  # 2^26 possible combinations
    'lives': 6,                  # 7 possible values
    'word_length': 9,            # 20+ possible values
}
# Total states: Massive! (impossible to enumerate)
```

**Solution:**
```python
# GOOD: Discretized/reduced state
state_key = f"{masked_word}:{len(guessed_letters)}:{lives}:{word_length}"
# Or use feature vector + function approximation (DQN)
```

**Parameters to Control:**
- **State discretization:** Simplify state representation
- **Use function approximation:** DQN instead of Q-table if state space is large
- **Limit word lengths:** Train on subset of lengths first (e.g., 4-12)

---

### Problem 2: Reward Design Challenges

**Issue:**
- Rewards too sparse → Agent learns slowly
- Rewards poorly balanced → Agent behaves suboptimally
- Wrong guess penalty too high → Agent too cautious
- Wrong guess penalty too low → Agent too reckless

**Example Bad Rewards:**
```python
# BAD: Sparse rewards
reward = 100 if won else -100 if lost else 0
# Agent only gets signal at end → learns very slowly

# BAD: Imbalanced rewards
reward = 1 if correct else -100 if wrong  # Too harsh!
# Agent becomes overly cautious, never explores
```

**Solution:**
```python
# GOOD: Balanced, informative rewards
rewards = {
    'correct_guess': +0.5,    # Immediate positive feedback
    'wrong_guess': -1.0,      # Penalty, but not too harsh
    'repeated_guess': -0.5,   # Discourage inefficiency
    'win': +10.0,             # Large bonus for success
    'lose': -5.0,             # Penalty, but less than win bonus
}
```

**Parameters to Tune:**
- **Correct guess reward:** 0.5 - 2.0 (immediate feedback)
- **Wrong guess penalty:** -0.5 - -2.0 (balance exploration)
- **Win bonus:** 10 - 50 (encourage completion)
- **Lose penalty:** -5 - -20 (less than win bonus)

**Tuning Strategy:**
- Start with moderate values
- Monitor agent behavior:
  - Too cautious? → Reduce wrong guess penalty
  - Too reckless? → Increase wrong guess penalty
  - Not finishing games? → Increase win bonus

---

### Problem 3: Action Space Issues

**Issue:**
- 26 possible actions (letters), but some are clearly bad
- Agent wastes time on low-probability letters
- Repeated guesses not properly handled

**Solution:**
```python
def get_available_actions(state):
    """Filter actions properly"""
    all_letters = set('abcdefghijklmnopqrstuvwxyz')
    guessed = state['guessed_letters']
    
    # Only return letters not yet guessed
    available = sorted(all_letters - guessed)
    
    # Optionally: Filter based on HMM probabilities
    # (don't allow letters with very low probability)
    # available = [a for a in available if hmm_probs[a] > threshold]
    
    return available
```

**Parameters:**
- **Action filtering:** Use HMM to filter obviously bad actions
- **Action masking:** Properly mask invalid actions in Q-values

---

### Problem 4: Episode Length Variation

**Issue:**
- Some words solved in 3 guesses, others need 10+
- Long episodes slow down training
- Short episodes may not learn long-term strategy

**Solution:**
```python
# Limit episode length
max_guesses = 20  # Cap total guesses
if num_guesses >= max_guesses:
    done = True
    reward = -10  # Penalize unfinished games
```

**Parameters:**
- **Max guesses per episode:** 15-25 (prevent infinite episodes)
- **Early termination:** Terminate if agent stuck

---

### Problem 5: Exploration vs Exploitation

**Issue:**
- Too much exploration → Agent wastes guesses on random letters
- Too little exploration → Agent gets stuck in local optimum
- ε-decay schedule not tuned properly

**Solution:**
```python
class QLearningAgent:
    def __init__(self, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.epsilon = epsilon           # Start exploring
        self.epsilon_decay = epsilon_decay  # Decay rate
        self.epsilon_min = epsilon_min    # Minimum exploration
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            # Exploration: use HMM probabilities (smart exploration)
            return self.explore_with_hmm(state)
        else:
            # Exploitation: use Q-values
            return self.exploit(state)
```

**Parameters to Tune:**
- **Initial epsilon:** 1.0 (100% exploration at start)
- **Epsilon decay:** 0.99 - 0.999 (how fast to reduce exploration)
  - 0.99: Decays fast → Exploit sooner
  - 0.999: Decays slow → Explore longer
- **Minimum epsilon:** 0.01 - 0.1 (maintain some exploration)

**Recommendation:**
- Start with ε=1.0, decay=0.995, min=0.01
- Monitor: If agent not exploring enough, increase min_epsilon
- Monitor: If agent too random, decrease decay slower

---

### Problem 6: State Representation Complexity

**Issue:**
- Too simple → Agent can't distinguish similar situations
- Too complex → Slow training, overfitting

**Solution - Start Simple, Then Improve:**

**Version 1: Simple String State**
```python
state_key = f"{masked_word}:{sorted(guessed_letters)}:{lives}"
# Example: "_PPLE:aep:5"
```

**Version 2: Feature Vector (For DQN)**
```python
def state_to_vector(state, hmm_probs):
    """Convert state to numerical vector"""
    features = []
    
    # Masked word encoding (one-hot for each position)
    for char in state['masked_word']:
        if char == '_':
            features.extend([1, 0])  # Is blank
        else:
            features.extend([0, 1])  # Is filled
    
    # Guessed letters (binary vector)
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features.append(1 if letter in state['guessed_letters'] else 0)
    
    # Lives left (normalized)
    features.append(state['lives_left'] / 6.0)
    
    # HMM probabilities
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features.append(hmm_probs.get(letter, 0))
    
    # Word length (normalized)
    features.append(state['word_length'] / 20.0)
    
    return np.array(features)
```

**Parameters:**
- **State encoding:** String (simple) vs Vector (complex)
- **Feature normalization:** Normalize to [0,1] for better training
- **Include HMM probs:** Yes! Very informative

---

### Problem 7: Different Word Lengths

**Issue:**
- Words of different lengths have very different optimal strategies
- Q-table can't share learning across lengths
- Training on all lengths at once is inefficient

**Solution:**
```python
# Option 1: Separate agents per length (if using Q-table)
agents_by_length = {
    4: QLearningAgent(),
    5: QLearningAgent(),
    # ... etc
}

# Option 2: Normalize positions (works with any approach)
def normalize_position(pos, word_length):
    return pos / word_length  # Position as fraction [0, 1]

# Option 3: Include word_length in state
state_key = f"{masked_word}:{word_length}:{lives}"
```

**Parameters:**
- **Length range:** Train on 4-15 letters initially (most common)
- **Length in state:** Include word_length in state representation
- **Separate agents:** Consider if using simple Q-learning

---

### Problem 8: Convergence Issues

**Issue:**
- Agent not converging to good policy
- Q-values oscillating
- Training loss not decreasing

**Solutions:**
```python
# Learning rate too high?
learning_rate = 0.1  # Start here, reduce if oscillating

# Discount factor too high?
discount_factor = 0.95  # 0.9-0.99 range

# Target network (for DQN)
# Update target network every N steps, not every step
```

**Parameters:**
- **Learning rate (α):** 0.01 - 0.1
  - Too high → Oscillations
  - Too low → Slow learning
- **Discount factor (γ):** 0.9 - 0.99
  - Higher → Values future rewards more
  - Lower → Focuses on immediate rewards
- **Target network update frequency (DQN):** Every 100-1000 steps

---

## 📋 Recommended Parameter Setup

### For Q-Learning (Table-based)

```python
class QLearningAgent:
    def __init__(self):
        # Learning parameters
        self.learning_rate = 0.1        # α: How fast to learn
        self.discount_factor = 0.95    # γ: Future reward importance
        
        # Exploration parameters
        self.epsilon = 1.0              # Start: 100% exploration
        self.epsilon_decay = 0.995      # Decay per episode
        self.epsilon_min = 0.01         # Minimum: 1% exploration
        
        # Q-table
        self.Q = defaultdict(lambda: defaultdict(float))
```

### For DQN (Deep Q-Network)

```python
class DQNAgent:
    def __init__(self):
        # Learning parameters
        self.learning_rate = 0.001      # Lower for neural networks
        self.discount_factor = 0.95
        
        # Exploration
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Network parameters
        self.batch_size = 32            # Replay buffer batch
        self.memory_size = 10000        # Replay buffer size
        self.target_update_freq = 100   # Update target network
        
        # Network architecture
        self.hidden_layers = [128, 64]  # Dense layers
```

### Environment Parameters

```python
class HangmanEnv:
    def __init__(self, word, max_lives=6, max_guesses=20):
        self.max_lives = max_lives        # Standard: 6
        self.max_guesses = max_guesses    # Prevent infinite: 20
        self.word = word.lower()
        
    # Rewards
    REWARDS = {
        'correct': +0.5,
        'wrong': -1.0,
        'repeated': -0.5,
        'win': +10.0,
        'lose': -5.0,
    }
```

---

## 🔧 Practical Implementation Template

```python
import numpy as np
from collections import defaultdict
import random

class HangmanEnv:
    """Hangman environment with recommended parameters"""
    
    def __init__(self, word, max_lives=6, max_guesses=25):
        self.word = word.lower()
        self.max_lives = max_lives
        self.max_guesses = max_guesses
        self.lives = max_lives
        self.guessed_letters = set()
        self.masked_word = ['_' for _ in self.word]
        self.num_guesses = 0
        
        # Reward parameters (tune these!)
        self.reward_correct = 0.5
        self.reward_wrong = -1.0
        self.reward_repeated = -0.5
        self.reward_win = 10.0
        self.reward_lose = -5.0
    
    def get_state(self):
        """Get state representation"""
        masked_str = ''.join(self.masked_word)
        return {
            'masked_word': masked_str,
            'guessed_letters': self.guessed_letters.copy(),
            'lives_left': self.lives,
            'word_length': len(self.word),
            'num_guesses': self.num_guesses
        }
    
    def state_to_key(self, hmm_probs=None):
        """Convert state to string key for Q-table"""
        state = self.get_state()
        
        # Option 1: Simple string (good for Q-table)
        guessed_str = ''.join(sorted(state['guessed_letters']))
        key = f"{state['masked_word']}:{guessed_str}:{state['lives_left']}"
        
        # Option 2: Include word length
        # key = f"{state['masked_word']}:{state['word_length']}:{guessed_str}:{state['lives_left']}"
        
        return key
    
    def guess_letter(self, letter):
        """Guess a letter, return (reward, new_state, done, info)"""
        letter = letter.lower()
        self.num_guesses += 1
        
        # Check repeated guess
        if letter in self.guessed_letters:
            reward = self.reward_repeated
            return reward, self.get_state(), False, {'status': 'repeated'}
        
        self.guessed_letters.add(letter)
        
        # Check if correct
        if letter in self.word:
            # Update masked word
            for i, char in enumerate(self.word):
                if char == letter:
                    self.masked_word[i] = letter
            
            # Check win
            if '_' not in self.masked_word:
                reward = self.reward_win
                return reward, self.get_state(), True, {'status': 'won'}
            else:
                reward = self.reward_correct
                return reward, self.get_state(), False, {'status': 'correct'}
        else:
            # Wrong guess
            self.lives -= 1
            
            if self.lives == 0 or self.num_guesses >= self.max_guesses:
                reward = self.reward_lose
                return reward, self.get_state(), True, {'status': 'lost'}
            else:
                reward = self.reward_wrong
                return reward, self.get_state(), False, {'status': 'wrong'}
    
    def reset(self, word=None):
        """Reset environment"""
        if word:
            self.word = word.lower()
        self.lives = self.max_lives
        self.guessed_letters = set()
        self.masked_word = ['_' for _ in self.word]
        self.num_guesses = 0
        return self.get_state()
```

---

## 🎯 Parameter Tuning Checklist

- [ ] **Learning rate:** Start 0.1, reduce if oscillating
- [ ] **Discount factor:** 0.95 is good starting point
- [ ] **Epsilon schedule:** 1.0 → 0.01, decay=0.995
- [ ] **Rewards:** Balanced (+0.5/-1.0/+10.0/-5.0)
- [ ] **State representation:** Start simple, add complexity if needed
- [ ] **Action space:** Properly filter available actions
- [ ] **Episode length:** Cap at 20-25 guesses
- [ ] **Word lengths:** Start with subset (4-12), expand later

---

## 💡 Pro Tips

1. **Start Simple:** Basic string state, simple rewards, Q-learning
2. **Monitor Training:** Plot Q-values, rewards, win rate over time
3. **Iterate Fast:** Test on 10 words first, then scale up
4. **Tune One Thing at a Time:** Don't change all parameters at once
5. **Use HMM Wisely:** HMM probabilities should guide, not replace RL
6. **Save Checkpoints:** Save agent after each epoch to compare

---

## 📊 Expected Issues Summary

| Issue | Symptom | Solution |
|-------|---------|----------|
| State space explosion | Memory error, slow training | Simplify state or use DQN |
| Poor convergence | Win rate not improving | Tune learning rate, rewards |
| Too cautious | High win rate but slow | Reduce wrong guess penalty |
| Too reckless | Low win rate | Increase wrong guess penalty |
| Stuck in local optimum | Stops improving | Increase exploration (ε) |
| Long episodes | Training very slow | Cap episode length |
| Different word lengths | Poor performance on some lengths | Include length in state |

---

Good luck! Start simple and iterate! 🚀

