import constants as const
from preprocessing_embs_loader import *
from utilities import serialize, deserialize

def create_word_embeddings(embedding_family, words_set):
    """
    It returns the matrix of the embeddings of the lemmas of the training set.
    And the dictionary that maps a word to the id of the vector in the embeddings matrix.
    This dictionary is then used to encode the dataset in another module.
    """

    # ##########
    # a, b = deserialize('embs_fast.dat')
    # return a, b
    # ##########

    model = None
    if embedding_family == const.EmbeddingsFamily.GOOGLE:
        model = load_google_vectors(const.PRE_TRAINED_EMB_DIR_GOOGLE)

    elif embedding_family == const.EmbeddingsFamily.GLOVE:
        model = load_glove_vectors(const.PRE_TRAINED_EMB_DIR_GLOVE)

    elif embedding_family == const.EmbeddingsFamily.FASTTEXT:
        model = load_fasttext_vectors(const.PRE_TRAINED_EMB_DIR_FASTTEXT)

    embeddings = []                         # it will contain the embedding vectors
    wemb_index = dict()                     # it maps a word to the id of the vector in the embeddings matrix
    words_set = list(words_set)             # the list of unique words in the training set
    words_set.insert(0, const.UNK_WORD_EMB)

    # Fill up the embedding matrix
    we_indx = 0
    for wd in words_set:

        if embedding_family == const.EmbeddingsFamily.GOOGLE and wd in model.vocab.keys():
            wemb_index[wd] = we_indx
            embeddings.append(model.wv[wd])
            we_indx += 1

        elif not embedding_family == const.EmbeddingsFamily.GOOGLE and wd in model.keys():
            wemb_index[wd] = we_indx
            embeddings.append(model[wd])
            we_indx += 1

    # serialize((np.array(embeddings), wemb_index), 'embs_fast.dat')
    # exit(1)

    return np.array(embeddings), wemb_index


def create_sense_embeddings(senses):
    """
    It returns the matrix of the embeddings of the senses of the training set.
    And the dictionary that maps a word to the id of the vector in the embeddings matrix.
    This dictionary is then used to encode the dataset in another module.
    """

    # #########
    # a, b = deserialize('embs_senses.dat')
    # return a, b
    # ##########

    data = {}
    embeddings = []        # it will contain the embedding vectors.
    semb_index = dict()    # it maps a word to the id of the vector in the embeddings matrix.
    se_indx = 0

    senses = list(senses)  # the list of unique senses in the training set
    senses.insert(0, const.UNK_WORD_EMB)

    # Read the file containing the sense vectors
    for filename in os.listdir(const.EmbeddingsFamily.SENSE.value):
        if filename.startswith('.'): continue

        print('File', filename)

        with open(const.EmbeddingsFamily.SENSE.value + "/" + filename, 'r') as fin:
            for line in fin:
                tokens = line.rstrip().split()
                try:
                    syn = tokens[0].split('_')[1] if len(tokens[0].split('_')) > 1 else tokens[0]   # BableSynset
                    embedding = np.array([float(val) for val in tokens[1:]])                        # Sense embedding

                    if len(embedding) == 400:
                        data[syn] = embedding
                except:
                    print('Error on:', tokens)
                    continue

        # Fill up the embedding matrix
        for word in senses:
            if word in data.keys():
                semb_index[word] = se_indx
                embeddings.append(data[word])
                se_indx += 1
                senses.remove(word)

        data = {}

    # serialize((np.array(embeddings), semb_index), 'embs_senses.dat')
    # exit(1)
    
    return np.array(embeddings), semb_index
