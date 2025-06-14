from typing import List, Dict

import nltk
import torch

from nltk.tokenize import word_tokenize

from src.data_utils.config import TextProcessorConfig


class TextProcessor:
    """
    Main text preprocessor class
    
    Args:
        vocab: Vocabulary dictionary
        config: Configuration object
    """

    def __init__(self, vocab: Dict[str, int], config: TextProcessorConfig):
        self.vocab = vocab
        self.config = config
        self._ensure_nltk_downloaded()


    def _ensure_nltk_downloaded(self):
        try:
            word_tokenize("test")
        except LookupError:
            nltk.download("punkt")
            nltk.download('punkt_tab')


    def preprocess_text(self, text: str) -> List[str]:
        """
        Tokenize and preprocess single text string

        Args: 
            text: Your text

        Returns:
            List of preprocessed tokens
        """

        if self.config.lowercase:
            text = text.lower()

        tokens = word_tokenize(text)

        if self.config.remove_punct:
            tokens = [t for t in tokens if t.isalpha()]

        return tokens


    def text_to_tensor(self, text: str) -> torch.Tensor:
        """
        Convert raw text to tensor
        
        Args: 
            text: Your text

        Returns:
            Tensor of your text
        """

        tokens = self.preprocess_text(text)
        ids = [self.vocab.get(token, self.vocab[self.config.unk_token]) for token in tokens]
        
        # Pad or truncate
        if len(ids) < self.config.max_seq_len:
            ids = ids + [self.vocab[self.config.pad_token]] * (self.config.max_seq_len - len(ids))
        else:
            ids = ids[:self.config.max_seq_len]
            
        return torch.tensor(ids, dtype=torch.long)
