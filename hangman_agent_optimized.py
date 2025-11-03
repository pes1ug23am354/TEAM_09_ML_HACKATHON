#!/usr/bin/env python3
"""
Hangman ML Agent - OPTIMIZED VERSION
Improved HMM with better letter prediction strategies
"""

import numpy as np
from collections import defaultdict, Counter
import random
import pickle

# Set random seeds
np.random.seed(42)
random.seed(42)

print("="*70)
print("HANGMAN ML AGENT - OPTIMIZED VERSION")
print("="*70)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

def preprocess_word_list(words, min_length=3, max_length=15):
    """Clean and filter word list"""
    words = list({word.lower() for word in words if word})
    return sorted([word for word in words 
                  if (min_length <= len(word) <= max_length and word.isalpha())])

print("\n[1/5] Loading data...")
with open('Data/corpus.txt', 'r', encoding='utf-8') as f:
    corpus_raw = [line.strip() for line in f if line.strip()]

with open('Data/test.txt', 'r', encoding='utf-8') as f:
    test_words_raw = [line.strip() for line in f if line.strip()]

corpus = preprocess_word_list(corpus_raw)
test_words = preprocess_word_list(test_words_raw)

print(f"  Corpus: {len(corpus)} words")
print(f"  Test: {len(test_words)} words")

# ============================================================================
# 2. IMPROVED HMM MODEL
# ============================================================================

class ImprovedHMM:
    """Improved HMM with multiple prediction strategies"""
    
    def __init__(self):
        self.letter_freq = Counter()
        self.bigram_freq = Counter()
        self.trigram_freq = Counter()
        self.start_letter_freq = Counter()
        self.end_letter_freq = Counter()
        self.position_freq = defaultdict(Counter)
        self.word_patterns = defaultdict(list)
    
    def train(self, corpus):
        """Train on corpus with multiple features"""
        print("\n[2/5] Training improved HMM...")
        
        for word in corpus:
            word_len = len(word)
            
            # Store word patterns by length
            self.word_patterns[word_len].append(word)
            
            # Letter frequencies
            for i, letter in enumerate(word):
                self.letter_freq[letter] += 1
                self.position_freq[word_len][i, letter] += 1
                
                if i == 0:
                    self.start_letter_freq[letter] += 1
                if i == len(word) - 1:
                    self.end_letter_freq[letter] += 1
            
            # Bigrams
            for i in range(len(word) - 1):
                self.bigram_freq[word[i:i+2]] += 1
            
            # Trigrams
            for i in range(len(word) - 2):
                self.trigram_freq[word[i:i+3]] += 1
        
        print(f"  Trained on {len(corpus)} words")
        print(f"  Unique bigrams: {len(self.bigram_freq)}")
        print(f"  Unique trigrams: {len(self.trigram_freq)}")
    
    def get_letter_scores(self, masked_word, guessed_letters):
        """Get letter scores using multiple strategies"""
        scores = defaultdict(float)
        word_len = len(masked_word)
        
        # Strategy 1: Match word patterns
        matching_words = []
        for word in self.word_patterns.get(word_len, []):
            if self._matches_pattern(word, masked_word, guessed_letters):
                matching_words.append(word)
        
        if matching_words:
            # Count letters in matching words
            letter_counts = Counter()
            for word in matching_words:
                for i, letter in enumerate(word):
                    if masked_word[i] == '_' and letter not in guessed_letters:
                        letter_counts[letter] += 1
            
            # Normalize
            total = sum(letter_counts.values())
            if total > 0:
                for letter, count in letter_counts.items():
                    scores[letter] += (count / total) * 10.0  # High weight
        
        # Strategy 2: Bigram/trigram patterns
        for i, char in enumerate(masked_word):
            if char == '_':
                # Check bigrams
                if i > 0 and masked_word[i-1] != '_':
                    prev = masked_word[i-1]
                    for letter in 'abcdefghijklmnopqrstuvwxyz':
                        if letter not in guessed_letters:
                            bigram = prev + letter
                            scores[letter] += self.bigram_freq[bigram] / 1000.0
                
                if i < len(masked_word) - 1 and masked_word[i+1] != '_':
                    next_char = masked_word[i+1]
                    for letter in 'abcdefghijklmnopqrstuvwxyz':
                        if letter not in guessed_letters:
                            bigram = letter + next_char
                            scores[letter] += self.bigram_freq[bigram] / 1000.0
                
                # Check trigrams
                if i > 0 and i < len(masked_word) - 1:
                    if masked_word[i-1] != '_' and masked_word[i+1] != '_':
                        prev = masked_word[i-1]
                        next_char = masked_word[i+1]
                        for letter in 'abcdefghijklmnopqrstuvwxyz':
                            if letter not in guessed_letters:
                                trigram = prev + letter + next_char
                                scores[letter] += self.trigram_freq[trigram] / 500.0
        
        # Strategy 3: Position-based frequencies
        for i, char in enumerate(masked_word):
            if char == '_':
                for letter in 'abcdefghijklmnopqrstuvwxyz':
                    if letter not in guessed_letters:
                        scores[letter] += self.position_freq[word_len][i, letter] / 100.0
        
        # Strategy 4: Overall letter frequency (fallback)
        total_freq = sum(self.letter_freq.values())
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            if letter not in guessed_letters:
                scores[letter] += (self.letter_freq[letter] / total_freq) * 0.5
        
        return scores
    
    def _matches_pattern(self, word, masked_word, guessed_letters):
        """Check if word matches the masked pattern"""
        if len(word) != len(masked_word):
            return False
        
        for i, (w_char, m_char) in enumerate(zip(word, masked_word)):
            if m_char != '_':
                if w_char != m_char:
                    return False
            else:
                if w_char in guessed_letters:
                    return False
        
        return True
    
    def get_best_guess(self, masked_word, guessed_letters):
        """Get best letter to guess"""
        scores = self.get_letter_scores(masked_word, guessed_letters)
        
        if not scores:
            # Fallback to most common letters
            common_letters = 'etaoinshrdlcumwfgypbvkjxqz'
            for letter in common_letters:
                if letter not in guessed_letters:
                    return letter
            return None
        
        # Return letter with highest score
        return max(scores.items(), key=lambda x: x[1])[0]

# ============================================================================
# 3. HANGMAN ENVIRONMENT
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
    
    def guess_letter(self, letter):
        """Make a guess"""
        letter = letter.lower()
        
        if letter in self.guessed_letters:
            self.repeated_guesses += 1
            return False, False
        
        self.guessed_letters.add(letter)
        
        if letter in self.word:
            for i, char in enumerate(self.word):
                if char == letter:
                    self.masked_word[i] = letter
            
            won = '_' not in self.masked_word
            return True, won
        else:
            self.lives -= 1
            self.wrong_guesses += 1
            lost = self.lives == 0
            return False, lost
    
    def get_masked_word(self):
        return ''.join(self.masked_word)

# ============================================================================
# 4. TRAINING AND EVALUATION
# ============================================================================

print("\n[3/5] Training HMM...")
hmm = ImprovedHMM()
hmm.train(corpus)

print("\n[4/5] Evaluating on test set...")
print(f"  Testing on {len(test_words)} words...")

wins = 0
total_wrong_guesses = 0
total_repeated_guesses = 0
game_results = []

for i, word in enumerate(test_words):
    env = HangmanEnv(word)
    
    while env.lives > 0 and '_' in env.masked_word:
        masked = env.get_masked_word()
        guess = hmm.get_best_guess(masked, env.guessed_letters)
        
        if guess is None:
            break
        
        correct, done = env.guess_letter(guess)
        
        if done:
            if correct:  # Won
                wins += 1
            break
    
    total_wrong_guesses += env.wrong_guesses
    total_repeated_guesses += env.repeated_guesses
    
    game_results.append({
        'word': word,
        'won': '_' not in env.masked_word,
        'wrong_guesses': env.wrong_guesses,
        'repeated_guesses': env.repeated_guesses
    })
    
    if (i + 1) % 500 == 0:
        print(f"  Progress: {i + 1}/{len(test_words)} games")

# ============================================================================
# 5. RESULTS
# ============================================================================

print("\n" + "="*70)
print("[5/5] FINAL RESULTS")
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

with open('evaluation_results_optimized.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"\n✅ Results saved to 'evaluation_results_optimized.pkl'")
print(f"✅ Evaluation complete!")
