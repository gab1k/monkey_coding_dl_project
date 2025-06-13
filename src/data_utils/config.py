from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """
    Configuration class for dataset generation parameters
    
    Attributes:
        embedding_dim: Dimension for embedding layer output
        train_size: Number of samples in training set
        val_size: Number of samples in validation set
        test_size: Number of samples in test set
        random_state: Random seed for reproducibility
        min_word_freq: Minimum word frequency to include in vocabulary
        load_from_disk: Load dataset from local dir. If false download from huggin face
        path_to_data: Path to local dataset data
        max_seq_len: Maximum sequence length (will be padded/truncated to this)
        lowercase: Whether to convert text to lowercase
        remove_punct: Whether to remove punctuation
        pad_token: Padding token
        unk_token: Unknown token    
    """

    embedding_dim: int = 64
    train_size: int = 10000
    val_size: int = 5000
    test_size: int = 5000
    random_state: int = 42
    min_word_freq: int = 1
    load_from_disk: bool = False
    path_to_data: str = "./datasets"

    max_seq_len: int = 300
    lowercase: bool = True
    remove_punct: bool = False
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"


@dataclass
class TextProcessorConfig:
    """
    Configuration class for text processor parameters (params should be equal dataset config)
    
    Attributes:
        max_seq_len: Maximum sequence length (will be padded/truncated to this)
        lowercase: Whether to convert text to lowercase
        remove_punct: Whether to remove punctuation
        pad_token: Padding token
        unk_token: Unknown token    
    """

    max_seq_len: int = 300
    lowercase: bool = True
    remove_punct: bool = False
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
