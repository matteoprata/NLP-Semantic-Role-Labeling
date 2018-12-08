from enum import Enum
import datetime as date

TRAIN_CORPUS = 'data/SRLData/EN/CoNLL2009-ST-English-train.txt'
DEV_CORPUS = 'data/SRLData/EN/CoNLL2009-ST-English-development.txt'
TEST_CORPUS = 'data/SRLData/EN/test.csv'

FIXED_COLS = 14

# Directores of pre-trained wordemeddings
PRE_TRAINED_EMB_DIR_GOOGLE = 'data/GoogleNews-vectors-negative300.bin'
PRE_TRAINED_EMB_DIR_FASTTEXT = 'data/wiki-news-300d-1M.vec'
PRE_TRAINED_EMB_DIR_GLOVE = 'data/glove'
PRE_TRAINED_SENSE_EMB = 'data/sense'

DISAMBIGUATED_TRAIN_DATA = 'data/disambiguated/train.txt'
DISAMBIGUATED_DEV_DATA = 'data/disambiguated/dev.txt'
DISAMBIGUATED_TEST_DATA = 'data/disambiguated/test.txt'

UNK_WORD_EMB = 'none'

MODEL_DIRECTORY = "mods/"


def get_date():
    return str(date.datetime.now().strftime("%Y%m%d%H%M%S"))


class EmbeddingsFamily(Enum):
    """
    The enumeration represents the possible embeddings matricies one can chose.
    """
    GLOVE = PRE_TRAINED_EMB_DIR_GLOVE
    GOOGLE = PRE_TRAINED_EMB_DIR_GOOGLE
    FASTTEXT = PRE_TRAINED_EMB_DIR_FASTTEXT
    SENSE = PRE_TRAINED_SENSE_EMB


class NeuralModel(Enum):
    """
    The enumeration represents the possible neural models one can chose.
    """
    ONE = 1
    TWO = 2
    THREE = 3
