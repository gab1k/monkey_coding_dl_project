"""
Dataset Generator Class for NLP tasks

This class provides a unified interface to generate tensor datasets for various NLP tasks.
It supports multiple datasets with configurable preprocessing parameters.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Tuple, List

import nltk
import pandas as pd
import torch

from datasets import load_dataset
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from torch import nn

import datasets_params


@dataclass
class DatasetConfig:
    """
    Configuration class for dataset generation parameters
    
    Attributes:
        max_seq_len: Maximum sequence length (will be padded/truncated to this)
        embedding_dim: Dimension for embedding layer output
        train_size: Number of samples in training set
        val_size: Number of samples in validation set
        test_size: Number of samples in test set
        random_state: Random seed for reproducibility
        min_word_freq: Minimum word frequency to include in vocabulary
        lowercase: Whether to convert text to lowercase
        remove_punct: Whether to remove punctuation
    """

    max_seq_len: int = 300
    embedding_dim: int = 64
    train_size: int = 10000
    val_size: int = 5000
    test_size: int = 5000
    random_state: int = 42
    min_word_freq: int = 1
    lowercase: bool = True
    remove_punct: bool = False


class DatasetGenerator:
    """
    Main dataset generator class
    
    Provides methods to load, preprocess and convert text datasets into tensor format
    suitable for deep learning models.
    
    Args:
        dataset_name: Name of dataset from DatasetName enum
        config: Configuration object with preprocessing parameters
        device: Torch device to place tensors on (cpu/cuda)
    """

    def __init__(
        self, 
        dataset_name: datasets_params.DatasetName,
        config: DatasetConfig = DatasetConfig(),
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.dataset_params = datasets_params.get_dataset_params_by_name(dataset_name=dataset_name)
        self.config = config
        self.device = device
        self.vocab = None
        self.id2word = None
        self.embedding_layer = None


    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load raw dataset from source
        
        Returns:
            Tuple of (train_df, val_df, test_df) DataFrames
        """

        dataset = load_dataset(self.dataset_params.hugging_face_name)
        train_df = pd.DataFrame(dataset["train"])
        test_df = pd.DataFrame(dataset["test"])
        val_df, test_df = train_test_split(
            test_df, 
            test_size=0.5, 
            random_state=self.config.random_state, 
            stratify=test_df[self.dataset_params.label_col_name]
        )
                    
        # Sample configured sizes
        train_df = train_df.sample(n=self.config.train_size, random_state=self.config.random_state)
        val_df = val_df.sample(n=self.config.val_size, random_state=self.config.random_state)
        test_df = test_df.sample(n=self.config.test_size, random_state=self.config.random_state)
        
        return train_df, val_df, test_df


    def preprocess_text(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize and preprocess text data
        
        Args:
            texts: List of raw text strings
            
        Returns:
            List of tokenized texts (list of tokens)
        """
        
        try:
            word_tokenize("test")
        except LookupError:
            nltk.download("punkt")

        tokenized_texts = []
        for text in texts:
            if self.config.lowercase:
                text = text.lower()
            tokens = word_tokenize(text)
            if self.config.remove_punct:
                tokens = [t for t in tokens if t.isalpha()]
            tokenized_texts.append(tokens)
            
        return tokenized_texts


    def build_vocabulary(self, tokenized_texts: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Build vocabulary from tokenized texts
        
        Args:
            tokenized_texts: List of tokenized texts
            
        Returns:
            Tuple of (word_to_id, id_to_word) mappings
        """

        all_tokens = [token for tokens in tokenized_texts for token in tokens]
        word_counts = Counter(all_tokens)
        
        # Filter by min frequency
        filtered_words = [word for word, count in word_counts.items() 
                         if count >= self.config.min_word_freq]
        
        # Create mappings
        word_to_id = {"<PAD>": 0, "<UNK>": 1}
        id_to_word = {0: "<PAD>", 1: "<UNK>"}
        
        for idx, word in enumerate(filtered_words, start=2):
            word_to_id[word] = idx
            id_to_word[idx] = word
            
        return word_to_id, id_to_word


    def text_to_tensor(
        self, 
        tokenized_texts: List[List[str]], 
        word_to_id: Dict[str, int]
    ) -> torch.Tensor:
        """
        Convert tokenized texts to tensor of word indices
        
        Args:
            tokenized_texts: List of tokenized texts
            word_to_id: Vocabulary mapping
            
        Returns:
            Tensor of shape (n_samples, max_seq_len)
        """

        sequences = []
        for tokens in tokenized_texts:
            # Convert tokens to ids, use UNK for unknown words
            ids = [word_to_id.get(token, word_to_id["<UNK>"]) for token in tokens]
            # Pad or truncate
            if len(ids) < self.config.max_seq_len:
                ids = ids + [word_to_id["<PAD>"]] * (self.config.max_seq_len - len(ids))
            else:
                ids = ids[:self.config.max_seq_len]
            sequences.append(ids)
            
        return torch.tensor(sequences, dtype=torch.long)


    def generate_embeddings(self, vocab_size: int) -> nn.Embedding:
        """
        Create embedding layer for the vocabulary
        
        Args:
            vocab_size: Size of vocabulary
            
        Returns:
            Initialized embedding layer
        """
        
        return nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.config.embedding_dim,
            padding_idx=0  # Assuming 0 is PAD index
        ).to(self.device)


    def generate_dataset(self) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        nn.Embedding
    ]:
        """
        Main method to generate the full dataset
        
        Returns:
            Tuple containing:
            - (train_features, train_labels)
            - (val_features, val_labels)
            - (test_features, test_labels)
            - embedding_layer
        """
        
        # Load raw data
        train_df, val_df, test_df = self.load_raw_data()
        
        # Preprocess text
        train_tokens = self.preprocess_text(train_df[self.dataset_params.content_col_name].tolist())
        val_tokens = self.preprocess_text(val_df[self.dataset_params.content_col_name].tolist())
        test_tokens = self.preprocess_text(test_df[self.dataset_params.content_col_name].tolist())
        
        # Build vocabulary from training data
        self.vocab, self.id2word = self.build_vocabulary(train_tokens)
        
        # Convert texts to tensors
        X_train = self.text_to_tensor(train_tokens, self.vocab)
        X_val = self.text_to_tensor(val_tokens, self.vocab)
        X_test = self.text_to_tensor(test_tokens, self.vocab)
        
        # Convert labels to tensors
        y_train = torch.tensor(train_df[self.dataset_params.label_col_name].values, dtype=torch.long)
        y_val = torch.tensor(val_df[self.dataset_params.label_col_name].values, dtype=torch.long)
        y_test = torch.tensor(test_df[self.dataset_params.label_col_name].values, dtype=torch.long)
        
        # Create embedding layer
        self.embedding_layer = self.generate_embeddings(len(self.vocab))
        
        # Apply embeddings to features
        with torch.no_grad():
            X_train = self.embedding_layer(X_train.to(self.device)).to(torch.float32)
            X_val = self.embedding_layer(X_val.to(self.device)).to(torch.float32)
            X_test = self.embedding_layer(X_test.to(self.device)).to(torch.float32)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), self.embedding_layer


    def get_vocabulary(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Get vocabulary mappings
        
        Returns:
            Tuple of (word_to_id, id_to_word) dictionaries
        """

        return self.vocab, self.id2word


    def get_config(self) -> DatasetConfig:
        """
        Get current configuration
        
        Returns:
            DatasetConfig object
        """

        return self.config
