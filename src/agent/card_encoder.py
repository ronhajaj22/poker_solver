import torch
import torch.nn as nn
import torch.nn.functional as F

class CardEncoder(nn.Module):
    """
    Neural network for encoding playing cards into dense vector representations.
    
    This encoder takes one-hot encoded cards (52-dimensional vectors) and transforms
    them into compact embeddings that capture card features like rank, suit, and
    relative strength. The network uses batch normalization and dropout for better
    training stability and generalization.
    """
    def __init__(self, embedding_dim=32, hidden_dim=64, dropout=0.1):
        """
        Initialize the card encoder network.
        
        Args:
            embedding_dim: Size of the output embedding vector (default: 32)
            hidden_dim: Size of hidden layers (default: 64)
            dropout: Dropout rate for regularization (default: 0.1)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Main neural network for card representation
        # This network learns to map one-hot encoded cards to meaningful embeddings
        self.card_net = nn.Sequential(
            # Input layer: Transform 52-dimensional one-hot encoding to hidden representation
            # 52 cards -> hidden_dim neurons
            nn.Linear(52, hidden_dim),
            # Batch normalization helps with training stability by normalizing activations
            # This is especially important when dealing with sparse one-hot inputs
            nn.BatchNorm1d(hidden_dim),
            # ReLU activation introduces non-linearity and helps with gradient flow
            nn.ReLU(),
            # Dropout randomly zeros some neurons during training to prevent overfitting
            nn.Dropout(dropout),
            
            # Hidden layer: Further compress the representation
            # hidden_dim -> hidden_dim//2 neurons (reducing dimensionality)
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output layer: Final projection to desired embedding dimension
            # hidden_dim//2 -> embedding_dim neurons
            nn.Linear(hidden_dim // 2, embedding_dim),
            # Tanh activation normalizes outputs to [-1, 1] range
            # This helps with numerical stability and makes embeddings comparable
            nn.Tanh()
        )
        
        # Optional: Separate specialized networks for rank and suit features
        # These can be used to explicitly model rank and suit relationships
        self.rank_encoder = nn.Sequential(
            # Encode 13 ranks (A, 2-10, J, Q, K) into compact representation
            nn.Linear(13, 16),  # 13 ranks -> 16 neurons
            nn.ReLU(),
            # Final rank embedding: 16 -> embedding_dim//4
            nn.Linear(16, embedding_dim // 4)
        )
        
        self.suit_encoder = nn.Sequential(
            # Encode 4 suits (Spades, Hearts, Diamonds, Clubs) into compact representation
            nn.Linear(4, 8),    # 4 suits -> 8 neurons
            nn.ReLU(),
            # Final suit embedding: 8 -> embedding_dim//4
            nn.Linear(8, embedding_dim // 4)
        )

    def forward(self, x, use_rank_suit=False):
        """
        Forward pass through the card encoder.
        
        Args:
            x: Input tensor of shape (batch_size, 52) - one-hot encoded cards
            use_rank_suit: If True, use separate rank/suit encoders instead of main network
        
        Returns:
            Tensor of shape (batch_size, embedding_dim) - card embeddings
        """
        if use_rank_suit and x.shape[1] == 52:
            # Alternative encoding method: explicitly extract rank and suit features
            # This can be useful when you want to model rank and suit relationships separately
            
            # Extract rank features from the one-hot encoding
            rank_features = self._extract_rank_features(x)
            # Extract suit features from the one-hot encoding
            suit_features = self._extract_suit_features(x)
            
            # Concatenate rank and suit features to form final embedding
            # This creates a structured representation where rank and suit are separate
            combined = torch.cat([rank_features, suit_features], dim=1)
            return combined
        
        # Standard encoding: use the main neural network
        return self.card_net(x)
    
    def _extract_rank_features(self, x):
        """
        Extract rank features from one-hot encoded cards.
        
        This method reshapes the 52-dimensional one-hot encoding to separate
        rank and suit information, then sums across suits to get rank presence.
        
        Args:
            x: Tensor of shape (batch_size, 52) - one-hot encoded cards
        
        Returns:
            Tensor of shape (batch_size, embedding_dim//4) - rank embeddings
        """
        # Reshape from (batch_size, 52) to (batch_size, 4, 13)
        # This separates the 52 cards into 4 suits × 13 ranks
        x_reshaped = x.view(x.shape[0], 4, 13)
        
        # Sum across suits (dim=1) to get rank presence
        # This tells us which ranks are present in the hand, regardless of suit
        rank_presence = torch.sum(x_reshaped, dim=1)  # Shape: (batch_size, 13)
        
        # Encode the rank presence into embeddings
        return self.rank_encoder(rank_presence)
    
    def _extract_suit_features(self, x):
        """
        Extract suit features from one-hot encoded cards.
        
        This method reshapes the 52-dimensional one-hot encoding to separate
        rank and suit information, then sums across ranks to get suit presence.
        
        Args:
            x: Tensor of shape (batch_size, 52) - one-hot encoded cards
        
        Returns:
            Tensor of shape (batch_size, embedding_dim//4) - suit embeddings
        """
        # Reshape from (batch_size, 52) to (batch_size, 4, 13)
        # This separates the 52 cards into 4 suits × 13 ranks
        x_reshaped = x.view(x.shape[0], 4, 13)
        
        # Sum across ranks (dim=2) to get suit presence
        # This tells us which suits are present in the hand, regardless of rank
        suit_presence = torch.sum(x_reshaped, dim=2)  # Shape: (batch_size, 4)
        
        # Encode the suit presence into embeddings
        return self.suit_encoder(suit_presence)

class CardAttentionEncoder(nn.Module):
    """
    Advanced card encoder with attention mechanism for handling multiple cards.
    
    This encoder uses self-attention to model relationships between cards in a hand.
    It's particularly useful for poker scenarios where card interactions matter.
    The attention mechanism allows the model to focus on relevant cards when
    encoding the entire hand.
    """
    def __init__(self, embedding_dim=32, num_heads=4):
        """
        Initialize the attention-based card encoder.
        
        Args:
            embedding_dim: Size of the output embedding vector (default: 32)
            num_heads: Number of attention heads (default: 4)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Initial card embedding layer
        # Maps each card's one-hot encoding to a dense representation
        self.card_embedding = nn.Linear(52, embedding_dim)
        
        # Multi-head self-attention mechanism
        # This allows the model to attend to different cards in the hand
        # num_heads=4 means the attention is split into 4 parallel attention mechanisms
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,  # Size of each card embedding
            num_heads=num_heads,      # Number of attention heads
            batch_first=True          # Input format: (batch, seq, features)
        )
        
        # Output projection layer
        # Further processes the attended representations
        self.output_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Layer normalization for training stability
        # Normalizes the activations after the residual connection
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        """
        Forward pass through the attention-based card encoder.
        
        Args:
            x: Input tensor of shape (batch_size, num_cards, 52)
               - batch_size: number of hands
               - num_cards: number of cards in each hand
               - 52: one-hot encoding for each card
        
        Returns:
            Tensor of shape (batch_size, num_cards, embedding_dim)
            - Each card gets an updated embedding that considers other cards
        """
        batch_size, num_cards, _ = x.shape
        
        # Step 1: Embed each card individually
        # Transform one-hot encodings to dense representations
        embeddings = self.card_embedding(x)  # Shape: (batch_size, num_cards, embedding_dim)
        
        # Step 2: Apply self-attention
        # Each card attends to all other cards in the hand
        # This allows cards to influence each other's representations
        attended, _ = self.attention(embeddings, embeddings, embeddings)
        
        # Step 3: Residual connection and layer normalization
        # Residual connection: add original embeddings to attended embeddings
        # This helps with gradient flow and preserves original card information
        attended = self.layer_norm(embeddings + attended)
        
        # Step 4: Final output projection
        # Further process the attended representations
        output = self.output_proj(attended)
        
        return output

def create_one_hot_card(card_index):
    """
    Create one-hot encoding for a single card.
    Args: card_index: Integer from 0-51 representing a card
    Returns: Tensor of shape (52,) with 1 at card_index position, 0 elsewhere
    """
    one_hot = torch.zeros(52)
    one_hot[card_index] = 1.0
    return one_hot

def create_hand_encoding(card_indices):
    """
    Create encoding for a hand of multiple cards.
    This function creates a multi-hot encoding where multiple cards
    can be present in the same hand (useful for poker hands).
    Args: card_indices: List of integers from 0-51 representing cards in the hand
    Returns: Tensor of shape (52) with 1s at positions corresponding to cards in hand
    """
    encoding = torch.zeros(52)
    for idx in card_indices:
        encoding[idx] = 1.0
    return encoding