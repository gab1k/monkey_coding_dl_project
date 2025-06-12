# HSE DL project spring-2025
## Литвинов Михаил и Габидуллин Камиль

### Запуск:
- `git clone https://github.com/gab1k/monkey_coding_dl_project.git`
- `cd monkey_coding_dl_project`
- Установка `poetry`, если нет - `pipx install poetry`
  - Надо установить `pipx`, если нет - `sudo apt install pipx; pipx ensurepath`
- Создание окружения и скачивание зависимостей - `poetry install`
- Запуск приложения на `Gradio` - `poetry run python main.py`
- Запуск ноутбуков - `poetry run jupyter notebook` и выбираем нужный

### Лейаут директорий:
- `./main.py` - точка входа в приложение на `Gradio`
- В директории `./src/app` ...
- В директории `./src/data` есть три файла 
  - `datasets_params.py` - содержит настройки использующихся датасетов [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb) и [fancyzhx/amazon_polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity)
  - `preprocessor.py` - содержит генератор датасетов - это класс, позволяющий делать всю обработку датасета одним методом
  - `datasets_stats.ipynb` - содержит примеры работ генератора и визуализацию статстик выбранных датасетов
- В директории `./src/models` ...
