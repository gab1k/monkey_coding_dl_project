import yaml

from dataclasses import dataclass


@dataclass
class AppConfig:
    model_path: str
    vocab_path: str
    config_path: str
    max_seq_len: int = 300
    share: bool = True
    host: str = "0.0.0.0"
    port: int = 7860

    @classmethod
    def from_yaml(cls, config_path: str) -> 'AppConfig':
        """
        AppConfig from path string

        Args:
            config_path: path string

        Returns:
            AppConfig object
        """
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls(
            model_path=config_data['model_path'],
            vocab_path=config_data['vocab_path'],
            config_path=config_data['config_path'],
            max_seq_len=int(config_data['max_seq_len']),
            share=config_data.get('server', {}).get('share', True),
            host=config_data.get('server', {}).get('host', "0.0.0.0"),
            port=config_data.get('server', {}).get('port', 7860)
        )
