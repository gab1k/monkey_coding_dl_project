import json
import torch
import nltk
from nltk.tokenize import word_tokenize
import argparse

from models import TransformerClassifier, MambaClassifier, LSTMClassifier

SAVE_DIR = "pretrained"
MODEL_PATH = f"{SAVE_DIR}/best_model.pth"
CONFIG_PATH = f"{SAVE_DIR}/config.json"
VOCAB_PATH = f"{SAVE_DIR}/vocab.json"

ID_TO_LABEL = {0: "Negative", 1: "Positive"}

def load_artifacts():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)

    model_type = config['model_type']
    model_params = config['model_params']

    if model_type == 'Transformer':
        model = TransformerClassifier(**model_params)
    elif model_type == 'Mamba':
        model = MambaClassifier(**model_params)
    elif model_type == 'LSTM':
        model = LSTMClassifier(**model_params)
    else:
        raise ValueError("Неизвестный тип модели в файле конфигурации.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, vocab, config, device

def preprocess_text(text, vocab, max_len):
    tokens = word_tokenize(text.lower())
    ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    if len(ids) < max_len:
        ids.extend([vocab['<PAD>']] * (max_len - len(ids)))
    else:
        ids = ids[:max_len]
    return torch.tensor(ids).unsqueeze(0)

def predict(text, model, vocab, config, device):
    input_tensor = preprocess_text(text, vocab, config['max_seq_len'])
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction_id = torch.argmax(probabilities, dim=1).item()
    
    predicted_label = ID_TO_LABEL[prediction_id]
    confidence = probabilities[0][prediction_id].item()

    return predicted_label, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Предсказать тональность текста с помощью обученной модели.")
    parser.add_argument("text", type=str, help="Текст для анализа (в кавычках).")
    args = parser.parse_args()

    print("Загрузка модели и артефактов...")
    try:
        loaded_model, loaded_vocab, loaded_config, device = load_artifacts()
        print(f"Модель '{loaded_config['model_type']}' успешно загружена на устройство {device}.")
    except FileNotFoundError:
        print("\nОШИБКА: Файлы модели не найдены!")
        print("Сначала запустите скрипт train.py для обучения и сохранения модели.")
        exit()

    label, conf = predict(args.text, loaded_model, loaded_vocab, loaded_config, device)

    print("\n--- Результат предсказания ---")
    print(f"Текст: '{args.text}'")
    print(f"Тональность: {label}")
    print(f"Уверенность: {conf:.2%}")

