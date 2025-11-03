#!/usr/bin/env python3
"""
Hangman ML Agent - Complete Training and Evaluation Script
Run this to train the HMM+RL agent and evaluate on 2000 test words
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import random
import pickle
import os
import re

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

print("="*70)
print("HANGMAN ML AGENT - TRAINING AND EVALUATION")
print("="*70)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

def preprocess_word_list(words, min_length=3, max_length=15):
    """Clean and filter word list"""
    # Convert to lowercase and remove duplicates
    words = list({word.lower() for word in words if word})
    # Filter words by length and letters only
    return sorted([word for word in words 
                  if (min_length <= len(word) <= max_length and 
                      word.isalpha())])

# Load data
print("\n[1/6] Loading data...")
corpus_path = 'Data/corpus.txt'
test_path = 'Data/test.txt'

with open(corpus_path, 'r', encoding='utf-8') as f:
    corpus_raw = [line.strip() for line in f if line.strip()]

with open(test_path, 'r', encoding='utf-8') as f:
    test_words_raw = [line.strip() for line in f if line.strip()]

print(f"  Raw corpus: {len(corpus_raw)} words")
print(f"  Raw test: {len(test_words_raw)} words")

# Preprocess
corpus = preprocess_word_list(corpus_raw)
test_words = preprocess_word_list(test_words_raw)

print(f"  Cleaned corpus: {len(corpus)} words")
print(f"  Cleaned test: {len(test_words)} words")

# Split into training and validation
split_idx = int(len(corpus) * 0.9)
training_corpus = corpus[:split_idx]
validation_corpus = corpus[split_idx:]

print(f"  Training: {len(training_corpus)} words")
print(f"  Validation: {len(validation_corpus)} words")

# ============================================================================
# 2. HANGMAN ENVIRONMENT
# ============================================================================

class HangmanEnv:
    """Hangman game environment"""
    
    def __init__(self, word, max_lives=6):
        self.word = word.lower()
        self.max_lives = max_lives
        self.lives = max_lives
        self.guessed_letters = set()
        self.masked_word = ['_' for _ in self.word]
        self.wrong_guesses = 0
        self.repeated_guesses = 0
    
    def get_state(self):
        """Get current state"""
        return {
            'masked_word': ''.join(self.masked_word),
            'guessed_letters': self.guessed_letters.copy(),
            'lives_left': self.lives,
            'word_length': len(self.word)
        }
    
    def guess_letter(self, letter):
        """Make a guess and return (reward, state, done, info)"""
        letter = letter.lower()
        
        # Check repeated guess
        if letter in self.guessed_letters:
            self.repeated_guesses += 1
            return -0.5, self.get_state(), False, {'status': 'repeated'}
        
        self.guessed_letters.add(letter)
        
        # Check if correct
        if letter in self.word:
            # Update masked word
            for i, char in enumerate(self.word):
                if char == letter:
                    self.masked_word[i] = letter
            
            # Check win
            if '_' not in self.masked_word:
                return 10.0, self.get_state(), True, {'status': 'won'}
            else:
                return 0.5, self.get_state(), False, {'status': 'correct'}
        else:
            # Wrong guess
            self.lives -= 1
            self.wrong_guesses += 1
            
            if self.lives == 0:
                return -5.0, self.get_state(), True, {'status': 'lost'}
            else:
                return -1.0, self.get_state(), False, {'status': 'wrong'}
    
    def reset(self, word=None):
        """Reset environment"""
        if word:
            self.word = word.lower()
        self.lives = self.max_lives
        self.guessed_letters = set()
        self.masked_word = ['_' for _ in self.word]
        self.wrong_guesses = 0
        self.repeated_guesses = 0
        return self.get_state()

# ============================================================================
# 3. HMM MODEL
# ============================================================================

class FirstOrderHMM:
    """First-order Hidden Markov Model for letter prediction"""
    
    def __init__(self, smoothing=0.01):
        self.smoothing = smoothing
        self.vocab_size = 26
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.start_counts = defaultdict(int)
        self.start_probs = defaultdict(float)
        self.letter_freq = defaultdict(int)
        self.total_letters = 0
    
    def train(self, corpus):
        """Train HMM on corpus"""
        print("\n[2/6] Training HMM...")
        
        for word in corpus:
            if len(word) == 0:
                continue
            
            for i, letter in enumerate(word):
                self.letter_freq[letter] += 1
                self.total_letters += 1
                
                if i == 0:
                    self.start_counts[letter] += 1
                else:
                    prev_letter = word[i-1]
                    self.transition_counts[prev_letter][letter] += 1
        
        # Compute probabilities with smoothing
        for prev_letter in self.transition_counts:
            total = sum(self.transition_counts[prev_letter].values())
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                count = self.transition_counts[prev_letter].get(letter, 0)
                self.transition_probs[prev_letter][letter] = (count + self.smoothing) / (total + self.smoothing * self.vocab_size)
        
        total_starts = sum(self.start_counts.values())
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            count = self.start_counts.get(letter, 0)
            self.start_probs[letter] = (count + self.smoothing) / (total_starts + self.smoothing * self.vocab_size)
        
        print(f"  Trained on {len(corpus)} words")
    
    def get_probabilities_for_mask(self, masked_word, guessed_letters=set()):
        """Get letter probabilities for current masked word"""
        letter_probs = defaultdict(float)
        
        # Find blank positions
        blank_positions = [i for i, char in enumerate(masked_word) if char == '_']
        
        if not blank_positions:
            return {letter: 0 for letter in 'abcdefghijklmnopqrstuvwxyz'}
        
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            if letter in guessed_letters:
                letter_probs[letter] = 0
                continue
            
            probs = []
            for pos in blank_positions:
                # Get previous letter
                prev_letter = None
                for i in range(pos - 1, -1, -1):
                    if masked_word[i] != '_':
                        prev_letter = masked_word[i]
                        break
                
                # Calculate probability
                if prev_letter is None:
                    prob = self.start_probs.get(letter, self.smoothing / self.vocab_size)
                else:
                    prob = self.transition_probs.get(prev_letter, {}).get(letter, self.smoothing / self.vocab_size)
                
                probs.append(prob)
            
            letter_probs[letter] = max(probs) if probs else 0
        
        # Normalize
        total = sum(letter_probs.values())
        if total > 0:
            letter_probs = {k: v/total for k, v in letter_probs.items()}
        
        return letter_probs

# ============================================================================
# 4. RL AGENT
# ============================================================================

class QLearningAgent:
    """Q-Learning agent for Hangman"""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=1.0):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.q_table = defaultdict(lambda: defaultdict(float))
    
    def get_state_key(self, state):
        """Convert state to hashable key"""
        return (state['masked_word'], tuple(sorted(state['guessed_letters'])), state['lives_left'])
    
    def choose_action(self, state, hmm_probs):
        """Choose action using epsilon-greedy + HMM guidance"""
        available_letters = [l for l in 'abcdefghijklmnopqrstuvwxyz' 
                           if l not in state['guessed_letters']]
        
        if not available_letters:
            return None
        
        # Epsilon-greedy
        if random.random() < self.epsilon:
            # Explore: use HMM probabilities
            if hmm_probs:
                hmm_letters = [(l, hmm_probs.get(l, 0)) for l in available_letters]
                hmm_letters.sort(key=lambda x: x[1], reverse=True)
                return hmm_letters[0][0] if hmm_letters else random.choice(available_letters)
            return random.choice(available_letters)
        else:
            # Exploit: use Q-values
            state_key = self.get_state_key(state)
            q_values = [(l, self.q_table[state_key][l]) for l in available_letters]
            q_values.sort(key=lambda x: x[1], reverse=True)
            return q_values[0][0]
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-values"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        if done:
            target = reward
        else:
            next_q_values = [self.q_table[next_state_key][l] 
                           for l in 'abcdefghijklmnopqrstuvwxyz' 
                           if l not in next_state['guessed_letters']]
            target = reward + self.gamma * max(next_q_values) if next_q_values else reward
        
        current_q = self.q_table[state_key][action]
        self.q_table[state_key][action] = current_q + self.lr * (target - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ============================================================================
# 5. TRAINING
# ============================================================================

print("\n[3/6] Initializing models...")
hmm = FirstOrderHMM(smoothing=0.01)
hmm.train(training_corpus)

agent = QLearningAgent(learning_rate=0.1, discount_factor=0.95, epsilon=1.0)

print("\n[4/6] Training RL agent...")
num_training_episodes = 1000
training_words = random.sample(training_corpus, min(num_training_episodes, len(training_corpus)))

for episode, word in enumerate(training_words):
    env = HangmanEnv(word)
    state = env.get_state()
    done = False
    
    while not done:
        # Get HMM probabilities
        hmm_probs = hmm.get_probabilities_for_mask(state['masked_word'], state['guessed_letters'])
        
        # Choose action
        action = agent.choose_action(state, hmm_probs)
        if action is None:
            break
        
        # Take action
        reward, next_state, done, info = env.guess_letter(action)
        
        # Update Q-values
        agent.update(state, action, reward, next_state, done)
        
        state = next_state
    
    # Decay epsilon
    agent.decay_epsilon()
    
    if (episode + 1) % 200 == 0:
        print(f"  Episode {episode + 1}/{num_training_episodes}, epsilon={agent.epsilon:.3f}")

print("  Training complete!")

# ============================================================================
# 6. EVALUATION
# ============================================================================

print("\n[5/6] Evaluating on test set...")
print(f"  Testing on {len(test_words)} words...")

wins = 0
total_wrong_guesses = 0
total_repeated_guesses = 0
game_results = []

for i, word in enumerate(test_words):
    env = HangmanEnv(word)
    state = env.get_state()
    done = False
    
    while not done:
        # Get HMM probabilities
        hmm_probs = hmm.get_probabilities_for_mask(state['masked_word'], state['guessed_letters'])
        
        # Choose action (greedy - no exploration)
        available_letters = [l for l in 'abcdefghijklmnopqrstuvwxyz' 
                           if l not in state['guessed_letters']]
        
        if not available_letters:
            break
        
        # Use HMM probabilities primarily
        if hmm_probs:
            hmm_letters = [(l, hmm_probs.get(l, 0)) for l in available_letters]
            hmm_letters.sort(key=lambda x: x[1], reverse=True)
            action = hmm_letters[0][0]
        else:
            action = available_letters[0]
        
        # Take action
        reward, next_state, done, info = env.guess_letter(action)
        state = next_state
    
    # Record results
    if info['status'] == 'won':
        wins += 1
    
    total_wrong_guesses += env.wrong_guesses
    total_repeated_guesses += env.repeated_guesses
    
    game_results.append({
        'word': word,
        'won': info['status'] == 'won',
        'wrong_guesses': env.wrong_guesses,
        'repeated_guesses': env.repeated_guesses
    })
    
    if (i + 1) % 500 == 0:
        print(f"  Progress: {i + 1}/{len(test_words)} games")

# ============================================================================
# 7. RESULTS
# ============================================================================

print("\n" + "="*70)
print("[6/6] FINAL RESULTS")
print("="*70)

success_rate = wins / len(test_words)
final_score = (success_rate * 2000) - (total_wrong_guesses * 5) - (total_repeated_guesses * 2)

print(f"\nGames Played: {len(test_words)}")
print(f"Games Won: {wins}")
print(f"Success Rate: {success_rate*100:.2f}%")
print(f"Total Wrong Guesses: {total_wrong_guesses}")
print(f"Total Repeated Guesses: {total_repeated_guesses}")
print(f"\n{'='*70}")
print(f"FINAL SCORE: {final_score:.2f}")
print(f"{'='*70}")

# Score breakdown
print(f"\nScore Breakdown:")
print(f"  Success Rate × 2000 = {success_rate * 2000:.2f}")
print(f"  Wrong Guesses × 5 = -{total_wrong_guesses * 5:.2f}")
print(f"  Repeated Guesses × 2 = -{total_repeated_guesses * 2:.2f}")
print(f"  {'='*40}")
print(f"  TOTAL = {final_score:.2f}")

# Save results
results = {
    'success_rate': success_rate,
    'total_wrong_guesses': total_wrong_guesses,
    'total_repeated_guesses': total_repeated_guesses,
    'final_score': final_score,
    'game_results': game_results
}

with open('evaluation_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"\n✅ Results saved to 'evaluation_results.pkl'")
print(f"✅ Training complete!")
