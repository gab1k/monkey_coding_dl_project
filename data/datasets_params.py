import enum


class DatasetName(enum.Enum):
    """
    Supported dataset names enumeration
    """
    
    IMDB = "imdb"
    POLARITY = "polarity"


class DatasetParams:
    """
    Abstarct class for dataset 
    """
    
    hugging_face_name = ""
    content_col_name = ""
    label_col_name = ""


def get_dataset_params_by_name(dataset_name: DatasetName) -> DatasetParams:
    if dataset_name == DatasetName.IMDB:
        return ImbdParams()
    if dataset_name == DatasetName.POLARITY:
        return PolarityParams()
    
    raise ValueError(f"Unsupported dataset: {dataset_name}")


class ImbdParams(DatasetParams):
    """
    IMDB dataset params class
    """
    
    hugging_face_name = "stanfordnlp/imdb"
    content_col_name = "text"
    label_col_name = "label"


class PolarityParams(DatasetParams):
    """
    POLARITY dataset params class
    """
    
    hugging_face_name = "fancyzhx/amazon_polarity"
    content_col_name = "content"
    label_col_name = "label"
