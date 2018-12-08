import numpy as np
import constants as const

def encode_dataset(dataset, vocabs):
    """
    Given a Dataset object and a list of dictionaries, this function produces a list of lists
    (the sentences) representing the encoded flat dataset. In fact each word in the Dataset gets encoded according to the
    ids withing the input dictionaries.
    """

    # Unpack dictionaries for encoding
    wemb_id, semb_id, role_id, pos_id, lex_id = vocabs

    # Lists of encoded sentences
    encsent_wids, encsent_sids, encsent_roles, encsent_posts, encsent_lexs, predicates = ([] for _ in range(6))

    for ind_s, sentence in enumerate(dataset.sentences):   # for each sentence
        for pred in sentence.predicates:                   # for each predicate of the sentence

            # Lists of encoded words
            x_wids, x_sids, y_roles, y_posts, y_lexs = ([] for _ in range(5))

            predicates.append(pred.id - 1)   # Position of the predicate in the sentence

            for wd in sentence.words:        # For each word in the sentence

                # Get the id of the lemma embedding of each word
                x_val = wemb_id.get(wd.lemma) if wemb_id.get(wd.lemma) else wemb_id.get(const.UNK_WORD_EMB)
                x_wids.append(x_val)

                # Get the role id of each word
                y_val = role_id.get(wd.get_role(pred.id)) if wd.get_role(pred.id) else role_id.get('NULL')
                y_roles.append(y_val)

                # Get the pos id of each word
                y_pos_val = pos_id.get(wd.pos) if pos_id.get(wd.pos) else pos_id.get('UNK')
                y_posts.append(y_pos_val)

                # Get the lex id of each word
                y_lex_val = lex_id.get(wd.lex) if lex_id.get(wd.lex) else lex_id.get('UNK')
                y_lexs.append(y_lex_val)

                # Get the id of the sense embedding of each word
                if not wd.sense == "-" and semb_id.get(wd.sense):
                    x_sense_val = semb_id.get(wd.sense)
                elif semb_id.get(wd.form.lower().strip()):
                    x_sense_val = semb_id.get(wd.form.lower().strip())
                else:
                    x_sense_val = semb_id.get(const.UNK_WORD_EMB)
                x_sids.append(x_sense_val)

            encsent_wids.append(x_wids)
            encsent_sids.append(x_sids)
            encsent_roles.append(y_roles)
            encsent_posts.append(y_posts)
            encsent_lexs.append(y_lexs)

    return (encsent_wids, encsent_sids, encsent_roles, encsent_posts, encsent_lexs, predicates)


def generate_batches(dataset_x, encsent_sids, dataset_y, y_post, y_lex, predicates_ids, batch_size, seq):
    """
    It is responsible for creating mini batches. Given the list of sentences, of roles, of senses, of pos, of lex,
    of predicate ids, it gets batch_size elements sequentially, depending on the sequence number. This function also pads
    with zeros the generated elements of the batch which size is less than the size of the maximum generated element.
    """

    # The range of elements to draw from the sequences
    range_accept = slice(seq * batch_size, (seq + 1) * batch_size)

    subdataset_x = dataset_x[range_accept]
    subdataset_y = dataset_y[range_accept]
    subdataset_post_y = y_post[range_accept]
    subdataset_lex_y = y_lex[range_accept]
    subdataset_sense_id = encsent_sids[range_accept]
    pred_ids = predicates_ids[range_accept]

    sent_length = list()
    for sent in subdataset_x:
        sent_length.append(len(sent))

    max_length = np.max(sent_length)

    # Start the padding phase
    padded_batch_x, padded_batch_x_senses, padded_batch_y, padded_batch_post_y, padded_batch_lex_y = ([] for _ in range(5))

    for i in range(batch_size):
        padded_batch_x.append(np.pad(subdataset_x[i], (0, max_length - len(subdataset_x[i])), 'constant', constant_values=(0, 0)))
        padded_batch_x_senses.append(np.pad(subdataset_sense_id[i], (0, max_length - len(subdataset_sense_id[i])), 'constant', constant_values=(0, 0)))

        if len(dataset_y) > 0:
            padded_batch_y.append(np.pad(subdataset_y[i], (0, max_length - len(subdataset_y[i])), 'constant', constant_values=(0, 0)))

        padded_batch_post_y.append(np.pad(subdataset_post_y[i], (0, max_length - len(subdataset_post_y[i])), 'constant', constant_values=(0, 0)))
        padded_batch_lex_y.append(np.pad(subdataset_lex_y[i], (0, max_length - len(subdataset_lex_y[i])), 'constant', constant_values=(0, 0)))

    return padded_batch_x, padded_batch_x_senses, padded_batch_y, padded_batch_post_y, padded_batch_lex_y, sent_length, onehot_predicates(pred_ids, np.max(sent_length))


def onehot_predicates(predicates_id, batch_max):
    """
    For each predicate id in 'predicates_id', it produces a vector containing 'batch_max'-1 zeros and a 1 in the position
    of the id of the predicate. The output of this function are used in the neural model to indicate what of the words are
    predicates and what are not.
    """

    id_predicate = []
    onehot_pred = []

    for i, id in enumerate(predicates_id):
        id_predicate.append([i, id])

        onehot_sent = [[0]]*batch_max    # [000000000...]
        onehot_sent[id] = [1]            # add a 1 in position id 
        onehot_pred.append(onehot_sent)  # e.g. the following sentence has a predicate in position 0: [100000000...]

    return (id_predicate, onehot_pred)
