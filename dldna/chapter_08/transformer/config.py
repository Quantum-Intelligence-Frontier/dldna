class TransformerConfig:
    def __init__(self):
        self.vocab_size = 30000              # Vocabulary size (total number of tokens)
        self.hidden_size = 512               # Embedding and model's hidden layer dimension size (d_model)
        self.num_hidden_layers = 6           # Number of encoder and decoder layers (N)
        self.num_attention_heads = 8         # Number of heads in multi-head attention (h)
        self.intermediate_size = 2048        # Hidden layer size of the Feed Forward Network (d_ff)
        self.dropout_prob = 0.1              # Default dropout probability
        self.hidden_dropout_prob = 0.1       # Dropout probability for hidden layers
        self.attention_probs_dropout_prob = 0.1  # Dropout probability for attention scores
        self.max_position_embeddings = 7     # Maximum length of position embeddings (3 digits + '+' + 3 digits)
        self.layer_norm_eps = 1e-6           # Epsilon value for layer normalization