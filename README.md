# HSE DL project spring-2025
## Литвинов Михаил и Габидуллин Камиль

### Предварительные шаги:
- `git clone https://github.com/gab1k/monkey_coding_dl_project.git`
- `cd monkey_coding_dl_project`
- Установка `poetry`, если нет - `pipx install poetry`
  - Надо установить `pipx`, если нет - `sudo apt install pipx; pipx ensurepath`
- Создание окружения и скачивание зависимостей - `poetry install`

### Настройка кофига приложения (закоммичен дефолтный вариант для локального запуска):
- `model_path` - 
- `vocab_path` -
- `config_path` -
- `server.local` - 
  - Если указано `true`, то `server.host` и `server.port` - хост и порт соответсвенно

### Запуск приложения/ноутбуков:
- Запуск приложения на `Gradio` - `poetry run python main.py`
- Запуск ноутбуков - `poetry run jupyter notebook` и выбираем нужный

### Лейаут директорий:
- `./config.yaml` - конфиг приложения
- `./main.py` - точка входа в приложение на `Gradio`
- В директории `./src/app` ...
- В директории `./src/data_utils` четыре файла:
  - `datasets_params.py` - содержит настройки использующихся датасетов [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb) и [fancyzhx/amazon_polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity)
  - `preprocessor.py` - содержит генератор датасетов - это класс, позволяющий делать всю обработку датасета одним методом
  - `datasets_stats.ipynb` - содержит примеры работ генератора и визуализацию статстик выбранных датасетов
- В директории `./src/models` ...
