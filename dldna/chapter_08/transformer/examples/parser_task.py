import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from typing import List, Tuple, Optional
import torch.nn.functional as F 

from typing import List, Tuple, Optional

from dldna.chapter_08.transformer.config import TransformerConfig
from dldna.chapter_08.transformer.transformer import Transformer
from dldna.chapter_08.transformer.masking import create_pad_mask, create_subsequent_mask

# Model save path setup
MODEL_PATH = "saved_models"
MODEL_FILE = "transformer_parser_task.pth"


def save_model(model: Transformer, config: TransformerConfig) -> None:
    """Save the model and configuration"""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': vars(config)
    }
    torch.save(save_dict, os.path.join(MODEL_PATH, MODEL_FILE))
    print(f"Model saved to {os.path.join(MODEL_PATH, MODEL_FILE)}")

def load_model() -> Tuple[Transformer, TransformerConfig]:
    """Load the saved model and configuration"""
    model_path = os.path.join(MODEL_PATH, MODEL_FILE)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model found at {model_path}")
    
    saved_dict = torch.load(model_path)
    
    config = TransformerConfig()
    for key, value in saved_dict['config'].items():
        setattr(config, key, value)
    
    model = Transformer(config)
    model.load_state_dict(saved_dict['model_state_dict'])
    
    return model, config


# Token dictionary definition
TOKEN_DICT = {
    'PAD': 0,
    '=': 1,
    '+': 2,
    '-': 3,
    '*': 4,
    '/': 5,
    'ASSIGN': 6,
    'ADD': 7,
    'SUB': 8,
    'MUL': 9,
    'DIV': 10,
    'x': 11,
    'y': 12,
    'z': 13
}

# Token range for numbers and variables
NUM_START = 13  # 0-9 are 13-22
VAR_START = 23  # a-z are 23-48

def generate_random_expression(max_tokens: int) -> str:
    """Generate a random expression"""
    ops = ['+', '-', '*', '/']
    vars = ['x', 'y', 'z']
    # Choose a variable
    var = np.random.choice(vars)
    # First number (0-9)
    num1 = str(np.random.randint(0, 10))
    # Choose an operator
    op = np.random.choice(ops)
    # Second number (0-9)
    num2 = str(np.random.randint(0, 10))
    # Create the expression
    expr = f"{var}={num1}{op}{num2}"
  
    return expr

def parse_to_tree(expr: str) -> List:
    """Convert an expression to a parse tree"""
    var, expr = expr.split('=')
    
    for op, op_token in [('+', 'ADD'), ('-', 'SUB'), ('*', 'MUL'), ('/', 'DIV')]:
        if op in expr:
            num1, num2 = expr.split(op)
            return ['ASSIGN', var, [op_token, num1, num2]]
    
    return ['ASSIGN', var, ['NUM', expr]]

def tokenize_expression(expr: str) -> List[int]:
    """Convert an expression to a sequence of tokens"""
    tokens = []
    for char in expr:
        if char in TOKEN_DICT:
            tokens.append(TOKEN_DICT[char])
        elif char.isdigit():
            tokens.append(int(char) + 14)  # Numbers start from 14
    return tokens

def tokenize_tree(tree: List) -> List[int]:
    """Convert a parse tree to a sequence of tokens"""
    tokens = []
    for item in tree:
        if isinstance(item, list):
            tokens.extend(tokenize_tree(item))
        else:
            if item in TOKEN_DICT:
                tokens.append(TOKEN_DICT[item])
            elif item.isdigit():
                tokens.append(int(item) + 14)  # Numbers start from 14
    return tokens

def decode_expression(tokens: List[int]) -> str:
    """Convert a sequence of tokens to an expression"""
    expr = []
    for token in tokens:
        if token in {v: k for k, v in TOKEN_DICT.items()}:
            expr.append({v: k for k, v in TOKEN_DICT.items()}[token])
        elif NUM_START <= token < VAR_START:
            expr.append(str(token - NUM_START))
        elif token >= VAR_START:
            expr.append(chr(ord('a') + (token - VAR_START)))
    return ' '.join(expr)

def decode_tree(tokens: List[int]) -> List:
    """Convert a sequence of tokens to a parse tree"""
    tree = []
    for token in tokens:
        if token in {v: k for k, v in TOKEN_DICT.items()}:
            tree.append({v: k for k, v in TOKEN_DICT.items()}[token])
        elif NUM_START <= token < VAR_START:
            tree.append(str(token - NUM_START))
        elif token >= VAR_START:
            tree.append(chr(ord('a') + (token - VAR_START)))
    return tree

def create_parser_data(batch_size: int = 32, max_tokens: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a parsing dataset"""
    expressions = []
    parse_trees = []
    
    # print("\n=== Start Generating Parsing Data ===")
    
    for i in range(batch_size):
        # Generate an expression
        expr = generate_random_expression(max_tokens)
        # print(f"\n[Batch {i+1}/{batch_size}]")
        # print(f"Generated Expression: {expr}")
        
        # Generate a parse tree
        tree = parse_to_tree(expr)
        # print(f"Parse Tree: {tree}")
        
        # Tokenize
        expr_tokens = tokenize_expression(expr)
        tree_tokens = tokenize_tree(tree)
        # print(f"Expression Tokens: {expr_tokens}")
        # print(f"Tree Tokens: {tree_tokens}")
        
        # Padding
        expr_tokens = expr_tokens + [TOKEN_DICT['PAD']] * (max_tokens - len(expr_tokens))
        tree_tokens = tree_tokens + [TOKEN_DICT['PAD']] * (max_tokens - len(tree_tokens))
        # print(f"Padded Expression Tokens: {expr_tokens}")
        # print(f"Padded Tree Tokens: {tree_tokens}")
        
        expressions.append(expr_tokens[:max_tokens])
        parse_trees.append(tree_tokens[:max_tokens])
    
    # print("\n=== Final Tensor Conversion ===")
    expressions_tensor = torch.tensor(expressions, dtype=torch.long)
    parse_trees_tensor = torch.tensor(parse_trees, dtype=torch.long)
    # print(f"Expression Tensor shape: {expressions_tensor.shape}")
    # print(f"Parse Tree Tensor shape: {parse_trees_tensor.shape}")
    
    return (expressions_tensor, parse_trees_tensor)

def show_parser_examples(num_examples: int = 5) -> None:
    """Function to show examples of parsing data generation"""
    print(f"\n=== Generating {num_examples} Parsing Examples ===\n")
    
    for i in range(num_examples):
        print(f"Example {i+1}:")
        # Generate an expression
        expr = generate_random_expression(max_tokens=5)
        print(f"Generated Expression: {expr}")
        
        # Generate a parse tree
        tree = parse_to_tree(expr)
        print(f"Parse Tree: {tree}")
        
        # Tokenize
        expr_tokens = tokenize_expression(expr)
        tree_tokens = tokenize_tree(tree)
        print(f"Expression Tokens: {expr_tokens}")
        print(f"Tree Tokens: {tree_tokens}")
        
        # Padding (fixed max_tokens=5)
        padded_expr = expr_tokens + [TOKEN_DICT['PAD']] * (5 - len(expr_tokens))
        padded_tree = tree_tokens + [TOKEN_DICT['PAD']] * (5 - len(tree_tokens))
        print(f"Padded Expression Tokens: {padded_expr[:5]}")
        print(f"Padded Tree Tokens: {padded_tree[:5]}")
        print()

def explain_parser_data(max_tokens: int = 5) -> None:
    """Explain the format and meaning of the generated parsing data"""
    src, target = create_parser_data(batch_size=1, max_tokens=max_tokens)
    
    expr = decode_expression(src[0].tolist())
    parsed = decode_tree(target[0].tolist())
    
    print("\n=== Parsing Data Explanation ===")
    print(f"Max Tokens: {max_tokens}")
    print("\n1. Input Sequence:")
    print(f"Original Tensor Shape: {src.shape}")
    print(f"Expression: {expr}")
    print(f"Tokenized Input: {src[0].tolist()}")
    print("\n2. Target Sequence:")
    print(f"Original Tensor Shape: {target.shape}")
    print(f"Parse Tree: {parsed}")
    print(f"Tokenized Output: {target[0].tolist()}")


def train_parser_task(
    config: TransformerConfig,
    num_epochs: int = 50,
    batch_size: int = 64,
    steps_per_epoch: int = 100,
    max_tokens: int = 5,
    print_progress: bool = True
) -> Transformer:
    """Train the parser model"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Transformer(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = ParserLoss().to(device)

    print("\n=== Start Training ===")
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Steps per Epoch: {steps_per_epoch}")
    print(f"Max Tokens: {max_tokens}\n")

    for epoch in range(num_epochs):
        model.train()
        total_loss = total_accuracy = 0

        for step in range(steps_per_epoch):
            # Data generation
            src, tgt = create_parser_data(batch_size, max_tokens)
            src, tgt = src.to(device), tgt.to(device)

            # Prepare decoder input (start token + target sequence)
            decoder_input = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            decoder_input = torch.cat([decoder_input, tgt[:, :-1]], dim=1)

            # Create masks
            src_mask = create_pad_mask(src).to(device)
            tgt_mask = create_subsequent_mask(decoder_input.size(1)).to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(src, decoder_input, src_mask, tgt_mask)

            # Loss calculation and backpropagation
            loss, accuracy = criterion(outputs, tgt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += accuracy

            # if print_progress and (step + 1) % steps_per_epoch == 0:
            #     _print_step_results(src, outputs, tgt, batch_size, max_tokens)

        # Print epoch results
        if epoch % 5 == 0:
            _print_epoch_summary(epoch, total_loss, total_accuracy, steps_per_epoch, optimizer)
            _print_step_results(src, outputs, tgt, batch_size, max_tokens)

    # Save the model
    save_model(model, config)
    return model

class ParserLoss(nn.Module):
    """Loss function for the parser task"""
    def __init__(self):
        super().__init__()
        self.step_counter = 0
    
    def forward(self, outputs: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, float]:
        batch_size = outputs.size(0)
        self.step_counter += 1
        
        predictions = F.softmax(outputs, dim=-1)
        target_one_hot = F.one_hot(target, num_classes=outputs.size(-1)).float()
        
        loss = -torch.sum(target_one_hot * torch.log(predictions + 1e-10)) / batch_size
        
        with torch.no_grad():
            pred_tokens = predictions.argmax(dim=-1)
            exact_match = (pred_tokens == target).all(dim=1).float()
            match_rate = exact_match.mean().item()
            
        return loss, match_rate

def _print_step_results(src, outputs, target, batch_size, max_tokens):
    """Print prediction results per step"""
    predictions = outputs.argmax(dim=-1)
    print("\n=== Prediction Result Samples ===")
    for i in range(min(2, batch_size)):
        expr = decode_expression(src[i].cpu().tolist())
        pred_tree = decode_tree(predictions[i].cpu().tolist())
        true_tree = decode_tree(target[i].cpu().tolist())
        match = "Correct" if pred_tree == true_tree else "Incorrect"
        print(f"Input: {expr}")
        print(f"Prediction: {pred_tree}")
        print(f"Truth: {true_tree}")
        print(f"Result: {match}\n")

def _print_epoch_summary(epoch: int, total_loss: float, total_accuracy: float, steps_per_epoch: int, optimizer) -> None:
    """Print epoch training results"""
    avg_loss = total_loss / steps_per_epoch
    avg_accuracy = total_accuracy / steps_per_epoch
    lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}, "
          f"Final Accuracy: {avg_accuracy:.4f}, "
          f"Learning Rate: {lr:.6f}")
    
def test_parser() -> None:
    """Test the trained parser model"""
    try:
        model, config = load_model()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Generate test data
        src, target = create_parser_data(batch_size=1, max_tokens=5)
        src, target = src.to(device), target.to(device)
        
        # Extract input expression
        expr = decode_expression(src[0].tolist())
        true_tree = decode_tree(target[0].tolist())
        
        src_mask = create_pad_mask(src).to(device)
        
        with torch.no_grad():
            decoder_input = torch.zeros((1, 1), dtype=torch.long, device=device)
            output_sequence = []
            
            # Generate 5 tokens (fixed length)
            for i in range(5):
                tgt_mask = create_subsequent_mask(decoder_input.size(1)).to(device)
                output = model(src, decoder_input, src_mask, tgt_mask)
                next_token = output[:, -1:].argmax(dim=-1)
                decoder_input = torch.cat([decoder_input, next_token], dim=1)
                output_sequence.append(next_token.item())
        
        predicted_tree = decode_tree(output_sequence)
        
        print("\n=== Parser Test ===")
        print(f"Input Expression: {expr}")
        print(f"Predicted Parse Tree: {predicted_tree}")
        print(f"Actual Parse Tree: {true_tree}")
        print(f"Correct: {predicted_tree == true_tree}")
        
        # Generate new expressions for additional testing
        print("\n=== Additional Tests ===")

        test_expressions = [
                    "x=1+2",
                    "y=3*4",
                    "z=5-1",
                    "x=2/3"
                ]
        
        for test_expr in test_expressions:
            # Tokenize the expression
            src_tokens = tokenize_expression(test_expr)
            src_tokens = src_tokens + [TOKEN_DICT['PAD']] * (5 - len(src_tokens))
            src = torch.tensor([src_tokens], dtype=torch.long).to(device)
            
            with torch.no_grad():
                decoder_input = torch.zeros((1, 1), dtype=torch.long, device=device)
                output_sequence = []
                
                for i in range(5):
                    tgt_mask = create_subsequent_mask(decoder_input.size(1)).to(device)
                    output = model(src, decoder_input, src_mask, tgt_mask)
                    next_token = output[:, -1:].argmax(dim=-1)
                    decoder_input = torch.cat([decoder_input, next_token], dim=1)
                    output_sequence.append(next_token.item())
            
            predicted_tree = decode_tree(output_sequence)
            print(f"\nInput: {test_expr}")
            print(f"Predicted Parse Tree: {predicted_tree}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Train the model first.")

def main():
    # if not os.path.exists(os.path.join(MODEL_PATH, MODEL_FILE)):
    #     print("=== Start Model Training ===")
    #     config = TransformerConfig()
    #     config.vocab_size = 50  # Adjust to match token dictionary size
    #     config.hidden_size = 128
    #     config.num_hidden_layers = 3
    #     config.num_attention_heads = 4
    #     config.intermediate_size = 256
    #     config.max_position_embeddings = 10
        
    #     model = train_parser_task(config, max_tokens=5)
        
    #     print("\n=== Parsing Test ===")
    #     test_parser(max_tokens=5)


    explain_parser_data()

if __name__ == "__main__":
    main()