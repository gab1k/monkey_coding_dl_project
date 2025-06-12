# HSE DL project spring-2025
## Литвинов Михаил и Габидуллин Камиль

### Предварительные шаги:
- `git clone https://github.com/gab1k/monkey_coding_dl_project.git`
- `cd monkey_coding_dl_project`
- Установка `poetry`, если нет - `pipx install poetry`
  - Надо установить `pipx`, если нет - `sudo apt install pipx; pipx ensurepath`
- Создание окружения и скачивание зависимостей - `poetry install`

### Настройка кофига приложения (закоммичен дефолтный вариант для локального запуска):
- `model_path` - путь к модели 
- `vocab_path` - путь к словарю
- `config_path` - путь к конфигу модели
- `max_seq_len` - максимальная длина контекста
- `server.local` - если указано `true`, то локальный запуск приложения, иначе - на сервере Hugging Face
  - Если указано `true`, то `server.host` и `server.port` - хост и порт соответсвенно

### Запуск приложения/ноутбуков:
- Запуск приложения с использованием `Gradio` - `poetry run python main.py`
- Запуск ноутбуков - `poetry run jupyter notebook` и выбираем нужный

### Лейаут директорий:
- Директория `./notebooks` содержит ноутбуки для различных задач
  - `datasets_stats.ipynb` - содержит примеры работ генератора и визуализацию статстик выбранных датасетов
  - `train.ipynb` ...
- Директория `./src/app` содержит класс приложения и все его компоненты:
  - Директория `model_utils` содержит фабрику и менеджер моделей
    - `factory.py` - фабрика для загрузки моделей
    - `manager.py` - инференс моделей
  - `app.py` - основной класс приложения
  - `config.py` - класс-обертка конфига приложения
- Директория `./src/data_utils` содержит классы для обработки датасетов:
  - `config.py` - содержит конфиги генератора и обработчика
  - `dataset_generator.py` - содержит генератор датасетов - это класс, позволяющий делать всю обработку датасета одним методом
  - `dataset_params.py` - содержит настройки использующихся датасетов [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb) и [fancyzhx/amazon_polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity)
  - `text_processor.py` - содержит обработчик текстов - это класс, позволяющий подготавливать тексты
- Директория `./src/models` ...
- `./config.yaml` - конфиг приложения
- `./main.py` - точка входа в приложение
