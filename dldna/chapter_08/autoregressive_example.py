import numpy as np

# Simple vocabulary
vocab = {
    0: "<START>",
    1: "I",
    2: "today",
    3: "study",
    4: "hard",
    5: "did",
    6: "<END>"
}

# Next word probability table (simple rule-based)
transition_probs = {
    0: [0, 1.0, 0, 0, 0, 0, 0],      # <START> -> "I" 100%
    1: [0, 0, 1.0, 0, 0, 0, 0],      # "I" -> "today" 100%
    2: [0, 0, 0, 1.0, 0, 0, 0],      # "today" -> "study" 100%
    3: [0, 0, 0, 0, 1.0, 0, 0],      # "study" -> "hard" 100%
    4: [0, 0, 0, 0, 0, 1.0, 0],      # "hard" -> "did" 100%
    5: [0, 0, 0, 0, 0, 0, 1.0],      # "did" -> <END> 100%
    6: [0, 0, 0, 0, 0, 0, 1.0]       # <END> -> stop generation
}

def demonstrate_autoregressive():
    """Visualizes the autoregressive sentence generation process."""
    sequence = [0]  # Start with <START>
    print("\n=== Autoregressive Generation Process Start ===")
    print(f"Start token: {vocab[sequence[0]]}")
    
    while True:
        # Print the sequence so far
        print("\nSentence generated so far:", " ".join([vocab[token] for token in sequence]))
        
        # Generate the next token based on the last token
        current_token = sequence[-1]
        next_token = np.random.choice(len(vocab), p=transition_probs[current_token])
        
        # Print next prediction token
        print(f"Next predicted word: {vocab[next_token]}")

        # Add the generated token
        sequence.append(next_token)
        
        # Stop generation if it's the END token
        if next_token == 6:
            break
    
    print("\n=== Final generated sentence ===")
    print(" ".join([vocab[token] for token in sequence[1:-1]]))  # Exclude START and END tokens
    
    return sequence

if __name__ == "__main__":
    demonstrate_autoregressive()