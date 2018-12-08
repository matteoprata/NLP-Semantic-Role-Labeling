from utilities import add_to_counter_dict
import constants as const
from preprocessing_dataset import *


def read_dataset(dataset_location):
    """
    Returns the dataset split by sentence, by word and by feature (columns of a words separated by \t).
    """

    sentences = []
    sentence = []

    with open(dataset_location, 'r') as f:
        whole_text = f.read()

    for word in whole_text.split('\n'):       # For each line in the dataset
        if not word == '':                    # If the sentence isn't over, add to 'sentence' the array of features of each word
            sentence.append(word.split('\t'))
        else:                                 # Else, add 'sentence' to 'sentences' and start a new sentence
            sentences.append(sentence)
            sentence = []

    return sentences[:-1]


def init_useful_dictionaries():

    # Set of unique words
    words_set = set()

    # Maps role tags - encoding
    role_id_dict = dict()
    role_id_dict['NULL'] = 0

    # Maps pos tags - encoding
    pos_id_dict = dict()
    pos_id_dict['UNK'] = 0

    # Maps lex tags - encoding
    lex_id_dict = dict()
    lex_id_dict['UNK'] = 0

    return words_set, role_id_dict, pos_id_dict, lex_id_dict


def create_dataset(sentences_in):
    """
    It parses the dataset and incapsulates its information into a Dataset object.
    :param sentences_in: a dataset split by sentence, by word and by feature.
    :return: a Dataset object and several dictionaries to keep information for the encodings.
    """

    words_set, role_id_dict, pos_id_dict, lex_id_dict = init_useful_dictionaries()

    dataset = Dataset()

    for sent_id, sent in enumerate(sentences_in):  # For each sentence (list of lines)

        n_roles_columns = len(sent[0]) - const.FIXED_COLS   # Number of columns of this sentence
        pred_counters = 0

        sentence = Sentence()

        for word in sent:
            wd_obj = Word()
            wd_obj.raw = "\t".join(word)        # the raw word, a line of the dataset

            wd_obj.sent_id = sent_id + 1        # id of the sentence of the word
            wd_obj.id = int(word[0])            # id of the word within the sentence
            wd_obj.form = word[1]               # the form of the word

            wd_obj.lemma = word[2]              # the lemma of the word
            words_set.add(wd_obj.lemma)         # add the lemma to the dictionary of unique words of the dataset

            wd_obj.pos = word[4]                   # the pos tag of the word
            wd_obj.lex = word[10]                  # the lexical information of the word
            wd_obj.is_predicate = word[12] == 'Y'  # true is the word is a predicate, false otherwise

            if wd_obj.is_predicate:
                sentence.add_predicate(wd_obj)     # add 'wd_obj' to the list of predicates of the sentence
                pred_counters += 1                 # count the number of predicates

            add_to_counter_dict(pos_id_dict, wd_obj.pos)  # ADD POS TAG TO 'pos_id_dict'
            add_to_counter_dict(lex_id_dict, wd_obj.lex)  # ADD LEX TAG TO 'lex_id_dict'

            sentence.add_word(wd_obj)  # add the Word object in the Sentence object

        sentence.n_predicates = pred_counters

        # -------------------- Now a new iteration over the words of 'sentence', so to add roles --------------------

        for wd_obj in sentence.words:
            dataset_word = sent[wd_obj.id - 1]  # a line of the input dataset

            for arg_column in range(const.FIXED_COLS, const.FIXED_COLS + n_roles_columns):  # For each column containing roles

                if not dataset_word[arg_column] == '_':                                     # If the word is an argument
                    predicate_id = sentence.predicates[arg_column - const.FIXED_COLS].id
                    role = dataset_word[arg_column]
                    w_instance = sentence.get_word(int(dataset_word[0]))

                    w_instance.set_predicate_role(predicate_id, role)            # set relation word:(predicate, role)
                    add_to_counter_dict(role_id_dict, role)                      # ADD ROLE TO 'role_id_dict'

        dataset.add_sentence(sentence)

    return dataset, words_set, role_id_dict, pos_id_dict, lex_id_dict


def disambiguate_dataset(dataset_location, dataset):
    """
    It adds to every word in the dataset, the BabelSynset id.
    :param dataset_location: Input file of the disambiguated dataset.
    :param dataset: the Dataset object to disambiguate.
    :return: a set of unique BabelSynset in the dataset.
    """
    senses = set()

    with open(dataset_location, 'r') as f:

        for sentence in dataset.sentences:
            for word in sentence.words:
                tokens = f.readline().strip().split('\t')
                word.sense = tokens[2]

                if not word.sense == '-':
                    senses.add(word.sense)
                else:
                    senses.add(word.lemma)

    return senses


def parse_dataset(corpus_dir):
    sentences = read_dataset(corpus_dir)                                                     # Read the input dataset
    dataset, words_set, role_id_dict, pos_id_dict, lex_id_dict = create_dataset(sentences)   # Parse the generated dataset

    senses = None

    # Disambiguate the dataset
    
    if corpus_dir == const.TRAIN_CORPUS:
        senses = disambiguate_dataset(const.DISAMBIGUATED_TRAIN_DATA, dataset)
    elif corpus_dir == const.DEV_CORPUS:
        senses = disambiguate_dataset(const.DISAMBIGUATED_DEV_DATA, dataset)
    elif corpus_dir == const.TEST_CORPUS:
        senses = disambiguate_dataset(const.DISAMBIGUATED_TEST_DATA, dataset)

    return dataset, (words_set, role_id_dict, pos_id_dict, lex_id_dict, senses)

