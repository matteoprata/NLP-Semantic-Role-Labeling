
from preprocessing_parser import parse_dataset
from preprocessing_vocabularies import *
from preprocessing_dataset_enc import *
import constants as const
from bilstm_srl import NeuralArchitecture


def get_datasets():

    # 1) Building training set, generate lemma and sense embeddings and the relative vocabularies
    dataset_train, (wd_id, role_id, pos_id, lex_id, sense_id_dict) = parse_dataset(const.TRAIN_CORPUS)
    wemb_matrix, wemb_id = create_word_embeddings(const.EmbeddingsFamily.FASTTEXT, wd_id)
    semb_matrix, semb_id = create_sense_embeddings(sense_id_dict)

    embeddings = wemb_matrix, semb_matrix
    vocabs = wemb_id, semb_id, role_id, pos_id, lex_id

    train_encodings = encode_dataset(dataset_train, vocabs)
    print('Done building training set.')

    # 2) Building development set
    dataset_dev, _ = parse_dataset(const.DEV_CORPUS)
    dev_encodings = encode_dataset(dataset_dev, vocabs)
    print('Done building development set.')

    # 3) Building test set
    dataset_test, _ = parse_dataset(const.TEST_CORPUS)
    test_encodings = encode_dataset(dataset_test, vocabs)
    print('Done building test set.')

    return train_encodings, dev_encodings, test_encodings, embeddings, vocabs, (dataset_test, role_id)


def run():
    """
    Given the training data, development data and test data, a neural architecture is executed over the data.
    """
    data = get_datasets()
    nn = NeuralArchitecture(data, const.NeuralModel.ONE)
    nn.execute_model()


run()