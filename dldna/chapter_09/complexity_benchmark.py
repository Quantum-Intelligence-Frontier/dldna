import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import psutil
import gc

def measure_mh_attention_complexity_gpu(seq_lengths: List[int], 
                              hidden_dim: int = 512,
                              num_heads: int = 8,
                              batch_size: int = 1) -> List[Tuple[int, float, float]]:
    """Measure memory usage and execution time of attention operation for various sequence lengths"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    
    for seq_length in seq_lengths:
        # Memory initialization
        torch.cuda.empty_cache()
        start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Input data generation
        q = torch.randn(batch_size, num_heads, seq_length, hidden_dim // num_heads, device=device)
        k = torch.randn(batch_size, num_heads, seq_length, hidden_dim // num_heads, device=device)
        v = torch.randn(batch_size, num_heads, seq_length, hidden_dim // num_heads, device=device)
        
        # Start measuring execution time
        start_time = time.time()
        
        # Perform attention operation
        # 1. Calculate QK^T
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(hidden_dim // num_heads)
        
        # 2. Apply Softmax
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        # 3. Calculate Attention(Q,K,V)
        attention_output = torch.matmul(attention_probs, v)
        
        # GPU synchronization (for accurate time measurement)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Calculate execution time
        elapsed_time = time.time() - start_time
        
        # Calculate memory usage
        end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_used = (end_mem - start_mem) / (1024 * 1024)  # MB unit
        
        results.append((seq_length, memory_used, elapsed_time))
        
        # Free memory
        del q, k, v, attention_scores, attention_probs, attention_output
        torch.cuda.empty_cache()
    
    return results

def measure_attention_complexity_gpu(seq_lengths: List[int], 
                              hidden_dim: int = 512,
                              batch_size: int = 1) -> List[Tuple[int, float, float]]:
    """Measure pure attention complexity with a single head."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    
    for seq_length in seq_lengths:
        torch.cuda.empty_cache()
        start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Use the entire dimension for a single head
        q = torch.randn(batch_size, seq_length, hidden_dim, device=device)
        k = torch.randn(batch_size, seq_length, hidden_dim, device=device)
        v = torch.randn(batch_size, seq_length, hidden_dim, device=device)
        
        start_time = time.time()
        
        # Pure attention operation
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(hidden_dim)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, v)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_used = (end_mem - start_mem) / (1024 * 1024)
        
        results.append((seq_length, memory_used, elapsed_time))
        
        del q, k, v, attention_scores, attention_probs, attention_output
        torch.cuda.empty_cache()
    
    return results

def measure_attention_complexity(seq_lengths: List[int], 
                              hidden_dim: int = 512,
                              batch_size: int = 32) -> List[Tuple[int, float, float]]:
    """Measure memory usage and execution time of attention operation on CPU"""
    device = torch.device("cpu")  # Use CPU
    results = []
    
    for seq_length in seq_lengths:
        # Input data generation
        q = torch.randn(batch_size, seq_length, hidden_dim, device=device)
        k = torch.randn(batch_size, seq_length, hidden_dim, device=device)
        v = torch.randn(batch_size, seq_length, hidden_dim, device=device)
        
        # Start measuring memory usage
        start_mem = psutil.Process().memory_info().rss / (1024 * 1024)  # MB unit
        
        # Start measuring execution time
        start_time = time.time()
        
        # Perform attention operation
        # 1. Calculate QK^T
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(hidden_dim)
        
        # 2. Apply Softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # 3. Calculate Attention(Q,K,V)
        attention_output = torch.matmul(attention_weights, v)
        
        # Calculate execution time
        elapsed_time = time.time() - start_time
        
        # Calculate memory usage
        end_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        memory_used = end_mem - start_mem
        
        results.append((seq_length, memory_used, elapsed_time))
        
        # Free memory
        del q, k, v, attention_scores, attention_weights, attention_output
        gc.collect()  # Force garbage collection
        
    return results


def plot_complexity_analysis(results: List[Tuple[int, float, float]]):
    """Visualize the results"""
    seq_lengths = [r[0] for r in results]
    memory_usage = [r[1] for r in results]
    time_taken = [r[2] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    
    # Memory usage graph
    ax1.plot(seq_lengths, memory_usage, 'b-o')
    ax1.set_title('Memory Usage vs Sequence Length')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Memory Usage (MB)')
    ax1.grid(True)
    
    # Execution time graph
    ax2.plot(seq_lengths, time_taken, 'r-o')
    ax2.set_title('Computation Time vs Sequence Length')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Time (seconds)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Run the experiment
    seq_lengths = [100, 500, 1000, 2000, 4000, 8000]
    results = measure_attention_complexity(seq_lengths)

    # Print the results
    print("\n=== Complexity Analysis of Attention Operation ===")
    print("\nMemory usage and execution time by sequence length:")
    print("Length\t\tMemory (MB)\tTime (seconds)")
    print("-" * 40)
    for seq_len, mem, time_taken in results:
        print(f"{seq_len}\t\t{mem:.2f}\t\t{time_taken:.4f}")

    # Visualize with a graph
    plot_complexity_analysis(results)

    # Compare theoretical complexity with actual measurements
    print("\n=== Comparison of Theoretical Complexity and Actual Measurements ===")
    base_seq = seq_lengths[0]
    base_mem = results[0][1]
    base_time = results[0][2]

    print("\nTheoretical vs Actual Growth Rate (Base: First Sequence Length)")
    print("Length\t\tTheoretical(N²)\tActual Memory\tActual Time")
    print("-" * 60)
    for seq_len, mem, time_taken in results:
        theoretical = (seq_len/base_seq) ** 2
        actual_mem = mem/base_mem
        actual_time = time_taken/base_time
        print(f"{seq_len}\t\t{theoretical:.2f}x\t\t{actual_mem:.2f}x\t\t{actual_time:.2f}x")