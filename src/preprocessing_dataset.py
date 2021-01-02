

class Word:
    """
    It represents a word of the dataset (a parsed line in the input file).
    A word is represented by the following components: the id; the id of the sentence; the word; the lemma;
    the part of speech tag; the lexical information; the BabelSynset; the line of the input file not parsed;
    if the word is a predicate; the predicates for which this word is an argument and the relative role.
    """

    id = None
    sent_id = None
    form = None
    lemma = None
    pos = None
    lex = None
    sense = None
    raw = None

    is_predicate = None
    predicate_role_dic = None

    def __init__(self):
        self.predicate_role_dic = {}

    def set_predicate_role(self, predicate_id, role):
        """ Each word knows the role it has with respect to a predicate. """
        self.predicate_role_dic[predicate_id] = role

    def get_role(self, predicate_id):
        """ Given the id of a predicate of the sentence of this word, it returns the role of this word.  """
        return self.predicate_role_dic.get(predicate_id)

    def __eq__(self, other):
        """ Two words are equal if their ids are the same.  """
        return self.sent_id == other.sent_id and self.id == other.id

    def __repr__(self):
        return "Word: id=" + str(self.sent_id) + "." + str(self.id) + " form=" + str(self.form) +\
               " lemma=" + str(self.lemma) + " pos=" + str(self.pos) + " is_predicate=" + str(self.is_predicate) +\
               " sense=" + str(self.sense) + "roles: " + str(self.predicate_role_dic)


class Sentence:
    """
    It represents a sentence of the dataset (a list of Word objects).
    A sentence is represented by the following components: the list of words (Word objects); a list of Word objects that
    are predicates of this sentence; and the number of predicates of this sentence.
    """

    words = None
    predicates = None
    n_predicates = None

    def __init__(self):
        self.words = []
        self.n_predicates = 0
        self.predicates = []

    def add_word(self, word):
        self.words.append(word)

    def add_predicate(self, word):
        self.predicates.append(word)

    def get_word(self, id):
        """ Given a word id withing the sentence, it returns the Word object associated to that id."""
        if 0 <= id < len(self.words):
            return self.words[id]
        return None

    def __repr__(self):
        out = '[\n'
        for w in self.words:
            out += str(w) + '\n'
        out += ']'
        return out


class Dataset:
    """
    It represents the list of sentences of the dataset (a list of Sentence objects).
    A dataset is represented by the following components: the list of sentences (Sentence objects).
    """

    sentences = None

    def __init__(self):
        self.sentences = []

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def __getitem__(self, item):
        try:
            return self.sentences[item]
        except Exception as error:
            print(error)

    def __repr__(self):
        out = ''
        for s in self.sentences:
            out += str(s) + '\n'
        return out.strip()

