#!/usr/bin/env python3
"""
Hangman ML Agent - FINAL OPTIMIZED VERSION
Uses smart word matching and frequency analysis
"""

import numpy as np
from collections import defaultdict, Counter
import random
import pickle

np.random.seed(42)
random.seed(42)

print("="*70)
print("HANGMAN ML AGENT - FINAL VERSION")
print("="*70)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

def preprocess_word_list(words, min_length=3, max_length=15):
    words = list({word.lower() for word in words if word})
    return [word for word in words if (min_length <= len(word) <= max_length and word.isalpha())]

print("\n[1/4] Loading data...")
with open('Data/corpus.txt', 'r', encoding='utf-8') as f:
    corpus = preprocess_word_list([line.strip() for line in f if line.strip()])

with open('Data/test.txt', 'r', encoding='utf-8') as f:
    test_words = preprocess_word_list([line.strip() for line in f if line.strip()])

print(f"  Corpus: {len(corpus)} words")
print(f"  Test: {len(test_words)} words")

# ============================================================================
# 2. SMART HANGMAN AGENT
# ============================================================================

class SmartHangmanAgent:
    """Smart agent using word matching and frequency analysis"""
    
    def __init__(self, corpus):
        self.corpus = corpus
        self.words_by_length = defaultdict(list)
        
        # Organize words by length
        for word in corpus:
            self.words_by_length[len(word)].append(word)
        
        # Precompute letter frequencies
        self.letter_freq = Counter(''.join(corpus))
        
        print(f"  Loaded {len(corpus)} words")
        print(f"  Word lengths: {min(self.words_by_length.keys())} to {max(self.words_by_length.keys())}")
    
    def get_matching_words(self, masked_word, guessed_letters):
        """Get all words that match the current pattern"""
        word_len = len(masked_word)
        candidates = self.words_by_length.get(word_len, [])
        
        matching = []
        for word in candidates:
            if self._matches_pattern(word, masked_word, guessed_letters):
                matching.append(word)
        
        return matching
    
    def _matches_pattern(self, word, masked_word, guessed_letters):
        """Check if word matches the masked pattern"""
        if len(word) != len(masked_word):
            return False
        
        for w_char, m_char in zip(word, masked_word):
            if m_char != '_':
                if w_char != m_char:
                    return False
            else:
                if w_char in guessed_letters:
                    return False
        
        return True
    
    def get_best_guess(self, masked_word, guessed_letters):
        """Get best letter to guess based on matching words"""
        # Get matching words
        matching_words = self.get_matching_words(masked_word, guessed_letters)
        
        if not matching_words:
            # Fallback to common letters
            common = 'etaoinshrdlcumwfgypbvkjxqz'
            for letter in common:
                if letter not in guessed_letters:
                    return letter
            return None
        
        # Count letter frequencies in matching words
        letter_counts = Counter()
        for word in matching_words:
            for i, letter in enumerate(word):
                if masked_word[i] == '_' and letter not in guessed_letters:
                    letter_counts[letter] += 1
        
        if not letter_counts:
            return None
        
        # Return most common letter
        return letter_counts.most_common(1)[0][0]

# ============================================================================
# 3. HANGMAN GAME
# ============================================================================

class HangmanGame:
    """Hangman game"""
    
    def __init__(self, word, max_lives=6):
        self.word = word.lower()
        self.max_lives = max_lives
        self.lives = max_lives
        self.guessed_letters = set()
        self.masked_word = ['_'] * len(word)
        self.wrong_guesses = 0
        self.repeated_guesses = 0
    
    def guess(self, letter):
        """Make a guess"""
        letter = letter.lower()
        
        if letter in self.guessed_letters:
            self.repeated_guesses += 1
            return 'repeated'
        
        self.guessed_letters.add(letter)
        
        if letter in self.word:
            for i, char in enumerate(self.word):
                if char == letter:
                    self.masked_word[i] = letter
            
            if '_' not in self.masked_word:
                return 'won'
            return 'correct'
        else:
            self.lives -= 1
            self.wrong_guesses += 1
            
            if self.lives == 0:
                return 'lost'
            return 'wrong'
    
    def get_masked_word(self):
        return ''.join(self.masked_word)
    
    def is_won(self):
        return '_' not in self.masked_word
    
    def is_lost(self):
        return self.lives == 0

# ============================================================================
# 4. TRAINING AND EVALUATION
# ============================================================================

print("\n[2/4] Initializing agent...")
agent = SmartHangmanAgent(corpus)

print("\n[3/4] Evaluating on test set...")
print(f"  Testing on {len(test_words)} words...")

wins = 0
total_wrong = 0
total_repeated = 0
results = []

for i, word in enumerate(test_words):
    game = HangmanGame(word)
    
    while not game.is_won() and not game.is_lost():
        masked = game.get_masked_word()
        guess = agent.get_best_guess(masked, game.guessed_letters)
        
        if guess is None:
            break
        
        status = game.guess(guess)
        
        if status in ['won', 'lost']:
            break
    
    if game.is_won():
        wins += 1
    
    total_wrong += game.wrong_guesses
    total_repeated += game.repeated_guesses
    
    results.append({
        'word': word,
        'won': game.is_won(),
        'wrong_guesses': game.wrong_guesses,
        'repeated_guesses': game.repeated_guesses,
        'lives_left': game.lives
    })
    
    if (i + 1) % 500 == 0:
        current_rate = wins / (i + 1)
        print(f"  Progress: {i + 1}/{len(test_words)} | Win rate: {current_rate*100:.1f}%")

# ============================================================================
# 5. FINAL RESULTS
# ============================================================================

print("\n" + "="*70)
print("[4/4] FINAL RESULTS")
print("="*70)

success_rate = wins / len(test_words)
final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)

print(f"\n📊 Performance Metrics:")
print(f"  Games Played: {len(test_words)}")
print(f"  Games Won: {wins}")
print(f"  Games Lost: {len(test_words) - wins}")
print(f"  Success Rate: {success_rate*100:.2f}%")
print(f"  Total Wrong Guesses: {total_wrong}")
print(f"  Total Repeated Guesses: {total_repeated}")
print(f"  Avg Wrong Guesses per Game: {total_wrong/len(test_words):.2f}")

print(f"\n💰 Score Calculation:")
print(f"  Success Rate × 2000 = {success_rate * 2000:.2f}")
print(f"  Wrong Guesses × 5 = -{total_wrong * 5:.2f}")
print(f"  Repeated Guesses × 2 = -{total_repeated * 2:.2f}")
print(f"  {'-'*40}")
print(f"  FINAL SCORE = {final_score:.2f}")

print(f"\n{'='*70}")
print(f"🎯 FINAL SCORE: {final_score:.2f}")
print(f"{'='*70}")

# Additional statistics
won_games = [r for r in results if r['won']]
lost_games = [r for r in results if not r['won']]

if won_games:
    avg_wrong_won = sum(r['wrong_guesses'] for r in won_games) / len(won_games)
    print(f"\n📈 Additional Stats:")
    print(f"  Avg wrong guesses (won games): {avg_wrong_won:.2f}")

if lost_games:
    avg_wrong_lost = sum(r['wrong_guesses'] for r in lost_games) / len(lost_games)
    print(f"  Avg wrong guesses (lost games): {avg_wrong_lost:.2f}")

# Save results
output = {
    'success_rate': success_rate,
    'total_wrong_guesses': total_wrong,
    'total_repeated_guesses': total_repeated,
    'final_score': final_score,
    'game_results': results
}

with open('final_results.pkl', 'wb') as f:
    pickle.dump(output, f)

print(f"\n✅ Results saved to 'final_results.pkl'")
print(f"✅ Evaluation complete!")
