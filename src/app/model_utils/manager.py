import json
import torch

from pathlib import Path
from typing import Dict, Any

from src.app.model_utils.factory import ModelFactory


class ModelManager:
    """
    Manages model loading and inference operations
    
    Args:
        model_dir: Directory containing model artifacts
    """
    
    def __init__(self, model_dir: str = "../pretrained") -> None:
        self.model_dir = Path(model_dir)
        self.loaded_models: Dict[str, Any] = {}
        self._load_model_artifacts()
        

    def _load_model_artifacts(self) -> None:
        """
        Load model configuration and vocabulary
        """

        with open(self.model_dir / "config.json", "r") as f:
            self.config = json.load(f)
        
        with open(self.model_dir / "vocab.json", "r") as f:
            self.vocab = json.load(f)
        
        self.idx_to_label = {0: "Negative", 1: "Positive"}
        

    def get_model(self) -> torch.nn.Module:
        """
        Get the loaded model (cached for performance)
        
        Returns:
            Loaded PyTorch model in evaluation mode
        """
        
        model_type = self.config["model_type"]
        
        if model_type not in self.loaded_models:
            model = ModelFactory.create_model(
                model_type=model_type,
                model_params=self.config["model_params"],
                state_dict_path=self.model_dir / "best_model.pth"
            )
            self.loaded_models[model_type] = model
        
        return self.loaded_models[model_type]
    

    def get_vocab(self) -> Dict[str, int]:
        """
        Get vocabulary mapping
        """

        return self.vocab
    

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration
        """
        
        return self.config
