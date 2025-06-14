import gradio as gr
import json
import torch

from typing import Optional

from src.app.config import AppConfig
from src.data_utils.config import TextProcessorConfig
from src.data_utils.text_processor import TextProcessor


class App:
    def __init__(self, config: AppConfig):
        self.config = config
        self.model: Optional[torch.nn.Module] = None
        self.text_processor: Optional[TextProcessor] = None
        
        self._load_model()
        self._load_text_processor()
    

    def _load_model(self):
        """
        Load model with params from config
        """

        with open(self.config.config_path, 'r') as f:
            config = json.load(f)
        
        model_type = config['model_type']
        model_classes = {
            'Transformer': 'TransformerClassifier',
            'LSTM': 'LSTMClassifier',
            'Mamba': 'MambaClassifier'
        }
        
        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
        
        module = __import__(f'src.models.models', fromlist=[model_classes[model_type]])
        model_class = getattr(module, model_classes[model_type])
        
        self.model = model_class(**config['model_params'])
        self.model.load_state_dict(torch.load(self.config.model_path))
        self.model.eval()


    def _load_text_processor(self):
        with open(self.config.vocab_path, 'r') as f:
            vocab = json.load(f)
        
        processor_config = TextProcessorConfig(
            max_seq_len=self.config.max_seq_len,
            lowercase=True,
            remove_punct=False
        )
        
        self.text_processor = TextProcessor(
            vocab=vocab,
            config=processor_config
        )
    

    def predict(self, text: str) -> dict:
        """
        Evaluating the tone of the text 
        """
        
        if not text.strip():
            return {"Negative": 0.5, "Positive": 0.5}
        
        input_tensor = self.text_processor.text_to_tensor(text).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            proba = torch.softmax(output, dim=1)[0].tolist()
        
        return {"Negative": proba[0], "Positive": proba[1]}


    def launch(self):
        """
        Launch interface
        """
        
        interface = gr.Interface(
            fn=self.predict,
            inputs=gr.Textbox(label="Enter your text"),
            outputs=gr.Label(label="Result"),
            title="Evaluating the tone of the text",
            examples=["Very good! Increadble! So fantastic", 
                    "Thw worst thing in the world!"]
        )
        
        if self.config.local:
            interface.launch(
                share=False,
                server_name=self.config.host,
                server_port=self.config.port
            )
        else:
            interface.launch(
                share=True
            )
