import torch

from pathlib import Path
from typing import Dict, Any, Optional

from src.models.models import TransformerClassifier, MambaClassifier, LSTMClassifier


class ModelFactory:
    """
    Factory class for creating and loading models
    """
    
    @staticmethod
    def create_model(
        model_type: str,
        model_params: Dict[str, Any],
        state_dict_path: Optional[Path] = None
    ) -> torch.nn.Module:
        """
        Create and load a model from configuration
        
        Args:
            model_type: Type of model ('Transformer', 'Mamba', 'LSTM')
            model_params: Dictionary of model parameters
            state_dict_path: Path to saved state dictionary
            
        Returns:
            Initialized PyTorch model
            
        Raises:
            ValueError: If model_type is unknown
        """

        model_classes = {
            "Transformer": TransformerClassifier,
            "Mamba": MambaClassifier,
            "LSTM": LSTMClassifier
        }
        
        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model_classes[model_type](**model_params)
        
        if state_dict_path:
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(state_dict)
        
        model.eval()
        return model
