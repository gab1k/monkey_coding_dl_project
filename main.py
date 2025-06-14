import warnings
for warn in [UserWarning, FutureWarning]: warnings.filterwarnings("ignore", category = warn)

from src.app.app import App
from src.app.config import AppConfig


def main():
    config = AppConfig.from_yaml("config.yaml")
    
    app = App(config)
    app.launch()


if __name__ == "__main__":
    main()
