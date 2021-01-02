import tensorflow as tf

def setup_model2(nn_architecture, embeddings_matrix, data_dictionaries):

    HIDDEL_LAYER_SIZE = 300  # The size of the hidden shared layer
    HIDDEN_SIZE = 50   # size of the latent representation of the context of a word
    BATCH_SIZE = 5     # size of each batch
    LR = 0.2           # learning rate for the optimizer
    EPOCHS = 20        # number of epochs

    N_ROLES = len(data_dictionaries[2])  # number of roles
    N_POS = len(data_dictionaries[3])    # number of pos tags
    N_LEX = len(data_dictionaries[4])    # number of lexical tags

    # ------------------------------------------- START TENSORFLOW GRAPH  ---------------------------------------------

    word_ids = tf.placeholder(tf.int32,   shape=[None, None])           # shape = [BATCH_SIZE, TIME-STEP]
    roles_ids = tf.placeholder(tf.int32,  shape=[None, None])           # shape = [BATCH_SIZE, TIME-STEP]
    senses_ids = tf.placeholder(tf.int32, shape=[None, None])           # shape = [BATCH_SIZE, TIME-STEP] #unused here
    post_ids = tf.placeholder(tf.int32,   shape=[None, None])           # shape = [BATCH_SIZE, TIME-STEP]
    lex_ids = tf.placeholder(tf.int32,    shape=[None, None])           # shape = [BATCH_SIZE, TIME-STEP]
    sequence_lengths = tf.placeholder(tf.int32, shape=[None])           # shape = [BATCH_SIZE]
    predicates_ids = tf.placeholder(tf.int32,   shape=[None, 2])        # shape = [BATCH_SIZE, 2]
    onehot_preds = tf.placeholder(tf.float32,   shape=[None, None, 1])  # shape = [BATCH_SIZE, TIME-STEP, 1]

    pretrained_lemma_embs = tf.Variable(embeddings_matrix[0], dtype=tf.float32, trainable=False)  # shape = [VOCAB-SIZE, LEMMA_EMBEDDINGS_SIZE]

    # The embeddings lookup translates encodings into embeddings.
    # A bit 1 or 0 is concatenated to the vector if it's a vector of a predicate
    lookedup_embs = tf.nn.embedding_lookup(pretrained_lemma_embs, word_ids)
    lookedup_embs = tf.concat([lookedup_embs, onehot_preds], axis=-1)

    encoding_lex = tf.one_hot([i for i in range(N_LEX)], N_LEX)
    encoding_lex = tf.nn.embedding_lookup(encoding_lex, lex_ids)

    encoding_pos = tf.one_hot([i for i in range(N_POS)], N_POS)
    encoding_pos = tf.nn.embedding_lookup(encoding_pos, post_ids)

    # WORLD REPRESENTATION: concatenation of the lemma embedding, a bit to identify the predicate, the lex and pos encoding
    # shape = [BATCH_SIZE, TIME-STEP, (LEMMA_EMBEDDINGS_SIZE + 1 + ENC_LEN_LEX + ENC_LEN_POS)]
    word_representation = tf.concat([lookedup_embs, encoding_lex, encoding_pos], axis=-1)

    # -------------------------------------------------- BILSTM (start) ----------------------------------------------

    cell_fw = tf.contrib.rnn.LSTMCell(num_units=HIDDEN_SIZE)  # LSTM layer 1
    cell_bw = tf.contrib.rnn.LSTMCell(num_units=HIDDEN_SIZE)  # LSTM layer 2

    (ofw, obw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, word_representation, sequence_length=sequence_lengths, dtype=tf.float32)

    context_rep = tf.concat([ofw, obw], axis=-1)  # shape = [BATCH_SIZE, TIME-STEP, 2*HIDDEN_SIZE]

    # -------------------------------------------------- BILSTM (end) ------------------------------------------------

    # A bit to identify the predicate is concatenated to the hidden representations
    # shape = [BATCH_SIZE, TIME-STEP, 2*HIDDEN_SIZE + 1]
    context_rep = tf.concat([context_rep, onehot_preds], axis=-1)

    # Get the hidden representations of the predicate for each sentence, multiply the predicate TIME-STEP times,
    # concatenate to each hidden representation, the hidden representation of the predicate.
    preds_context_rep = tf.gather_nd(context_rep, indices=predicates_ids)
    repl_preds_context = tf.tile(tf.expand_dims(preds_context_rep, axis=1), multiples=[1, tf.shape(context_rep)[1], 1])

    # shape = [BATCH_SIZE, TIME-STEP, (2*HIDDEN_SIZE) * 2 + 2]
    to_classify = tf.concat([context_rep, repl_preds_context], axis=-1)

    # shape = [N_WORDS, (2*HIDDEN_SIZE) * 2 + 2]
    to_classify_flat = tf.reshape(to_classify, [-1, (2 * HIDDEN_SIZE) * 2 + 2])

    # -------------------------------------------- CLASSIFIER  ------------------------------------------------------

    W = tf.get_variable("W", shape=[(2 * HIDDEN_SIZE) * 2 + 2, HIDDEL_LAYER_SIZE], dtype=tf.float32)
    Y1_W = tf.get_variable("Y1W", shape=[HIDDEL_LAYER_SIZE, N_ROLES], dtype=tf.float32)
    Y2_W = tf.get_variable("Y2W", shape=[HIDDEL_LAYER_SIZE, N_LEX], dtype=tf.float32)
    Y3_W = tf.get_variable("Y3W", shape=[HIDDEL_LAYER_SIZE, N_POS], dtype=tf.float32)

    predictions_shared = tf.matmul(to_classify_flat, W)   # shape = [N_WORDS, HIDDEL_LAYER_SIZE]
    predictions_Y1 = tf.matmul(predictions_shared, Y1_W)  # shape = [HIDDEL_LAYER_SIZE, N_ROLES]
    predictions_Y2 = tf.matmul(predictions_shared, Y2_W)  # shape = [HIDDEL_LAYER_SIZE, N_LEX]
    predictions_Y3 = tf.matmul(predictions_shared, Y3_W)  # shape = [HIDDEL_LAYER_SIZE, N_POS]

    predictions_scores_Y1 = tf.reshape(predictions_Y1, [-1, tf.shape(to_classify)[1], N_ROLES])  # shape = [BATCH_SIZE, TIMESTEP, N_ROLES]
    predictions_scores_Y2 = tf.reshape(predictions_Y2, [-1, tf.shape(to_classify)[1], N_LEX])    # shape = [BATCH_SIZE, TIMESTEP, N_LEX]
    predictions_scores_Y3 = tf.reshape(predictions_Y3, [-1, tf.shape(to_classify)[1], N_POS])    # shape = [BATCH_SIZE, TIMESTEP, N_POS]

    predictions_probabilities = tf.nn.softmax(predictions_scores_Y1)

    # Measure the prediction error
    losses_roles = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions_scores_Y1, labels=roles_ids)
    losses_lex = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions_scores_Y2, labels=lex_ids)
    losses_pos = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions_scores_Y3, labels=post_ids)

    joint_loss = losses_roles + losses_lex + losses_pos

    loss = tf.reduce_mean(tf.boolean_mask(joint_loss, tf.sequence_mask(sequence_lengths)))

    train = tf.train.AdagradOptimizer(LR).minimize(loss)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(max_to_keep=25)

    # ---------------
    # Make tensors and operations available to the NeuralArchitecture class
    # ---------------

    nn_architecture.EPOCHS = EPOCHS
    nn_architecture.BATCH_SIZE = BATCH_SIZE

    nn_architecture.word_ids = word_ids
    nn_architecture.roles_ids = roles_ids
    nn_architecture.senses_ids = senses_ids
    nn_architecture.post_ids = post_ids
    nn_architecture.lex_ids = lex_ids
    nn_architecture.predicates_ids = predicates_ids
    nn_architecture.sequence_lengths = sequence_lengths
    nn_architecture.onehot_preds = onehot_preds

    nn_architecture.train = train
    nn_architecture.loss = loss
    nn_architecture.predictions_probabilities = predictions_probabilities
    nn_architecture.saver = saver


