from collections import Counter
from typing import Dict, Tuple, List

import pandas as pd
import torch

from datasets import load_dataset, load_from_disk
from sklearn.model_selection import train_test_split

import src.data_utils.dataset_params as dataset_params

from src.data_utils.config import DatasetConfig, TextProcessorConfig
from src.data_utils.text_processor import TextProcessor


class DatasetGenerator:
    """
    Main dataset generator class
    
    Provides methods to load, build vocabulary, convert text datasets 
    into tensor format suitable for deep learning models.
    
    Args:
        dataset_name: Name of dataset from DatasetName enum
        config: Configuration object with preprocessing parameters
        device: Torch device to place tensors on (cpu/cuda)
    """

    def __init__(
        self, 
        dataset_name: dataset_params.DatasetName,
        config: DatasetConfig = DatasetConfig(),
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.dataset_params = dataset_params.get_dataset_params_by_name(dataset_name=dataset_name)
        self.config = config
        self.device = device
        self.text_processor = TextProcessor(
            vocab=None, 
            config=TextProcessorConfig(
                max_seq_len=self.config.max_seq_len,
                lowercase=self.config.lowercase,
                remove_punct=self.config.remove_punct,
                pad_token=self.config.pad_token,
                unk_token=self.config.unk_token,
            )
        )
        self.vocab = None
        self.id2word = None
        self.embedding_layer = None


    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load raw dataset from source
        
        Returns:
            Tuple of (train_df, val_df, test_df) DataFrames
        """
        if self.config.load_from_disk:
            dataset = load_from_disk(f"{self.config.path_to_data}/{self.dataset_params.local_path}")
        else:
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
        
        filtered_words = [word for word, count in word_counts.items() 
                         if count >= self.config.min_word_freq]
        
        word_to_id = {self.config.pad_token: 0, self.config.unk_token: 1}
        id_to_word = {0: self.config.pad_token, 1: self.config.unk_token}
        
        for idx, word in enumerate(filtered_words, start=2):
            word_to_id[word] = idx
            id_to_word[idx] = word
        
        self.text_processor.vocab = word_to_id

        return word_to_id, id_to_word


    def generate_dataset(self) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor]
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
        
        train_df, val_df, test_df = self.load_raw_data()

        train_texts = train_df[self.dataset_params.content_col_name].tolist()
        train_tokens = [self.text_processor.preprocess_text(text) for text in train_texts]
        
        self.vocab, self.id2word = self.build_vocabulary(train_tokens)

        X_train = torch.stack([self.text_processor.text_to_tensor(text) for text in train_texts])
        
        val_texts = val_df[self.dataset_params.content_col_name].tolist()
        X_val = torch.stack([self.text_processor.text_to_tensor(text) for text in val_texts])
        
        test_texts = test_df[self.dataset_params.content_col_name].tolist()
        X_test = torch.stack([self.text_processor.text_to_tensor(text) for text in test_texts])
        
        y_train = torch.tensor(train_df[self.dataset_params.label_col_name].values, dtype=torch.long)
        y_val = torch.tensor(val_df[self.dataset_params.label_col_name].values, dtype=torch.long)
        y_test = torch.tensor(test_df[self.dataset_params.label_col_name].values, dtype=torch.long)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


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


    def get_text_processor(self) -> TextProcessor:
        """
        Get the text processor for inference usage
        
        Returns:
            TextProcessor object
        """
        return self.text_processor
