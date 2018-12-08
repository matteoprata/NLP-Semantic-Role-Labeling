import tensorflow as tf
from tqdm import tqdm
from preprocessing_dataset_enc import generate_batches
import numpy as np
from constants import NeuralModel, MODEL_DIRECTORY, get_date
from utilities import reverse_dict

class NeuralArchitecture:

    train_data = None
    dev_data = None
    test_data = None
    dataset_test = None

    model = None                # model of neural architecture to run
    embeddings_matrices = None  # lemma embedding (index 0), sense embeddings (index 1)
    data_vocabularies = None    # vocabularies of the dataset (eg. lemmas, roles, senses...)

    EPOCHS = None               # number of epoch of the model
    BATCH_SIZE = None           # batch size of the model

    word_ids, roles_ids, senses_ids, post_ids, lex_ids, predicates_ids, sequence_lengths, onehot_preds = (None,)*8   # 8 placeholders to fill
    train, loss, predictions_probabilities, saver = (None,) * 4                                                      # 4 operations to run


    def __init__(self, data, model):
        """
        The neural architecture is built from the set of datasets needed for the computation and the name of the
        neural model to execute among the available ones.
        """

        train_data, dev_data, test_data, embeddings_matrix, data_dictionaries, dataset_test = data
        self.model = model

        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.dataset_test = dataset_test

        self.embeddings_matrices = embeddings_matrix
        self.data_vocabularies = data_dictionaries

    def build_graph(self):
        """
        It is responsible for building the TensorFlow graph of the desired model.
        """

        if self.model == NeuralModel.ONE:
            from model1 import setup_model1
            setup_model1(self, self.embeddings_matrices, self.data_vocabularies)

        elif self.model == NeuralModel.TWO:
            from model2 import setup_model2
            setup_model2(self, self.embeddings_matrices, self.data_vocabularies)

        elif self.model == NeuralModel.THREE:
            from model3 import setup_model3
            setup_model3(self, self.embeddings_matrices, self.data_vocabularies)

    def training(self, session):
        """
        It is responsible for training the neural model stored in the input session.
        """

        # Unpack training data: encoded words, senses, roles, post tags, lexical tags and the indicies of the predicates for each sentence.
        encsent_wids, encsent_sids, encsent_roles, encsent_posts, encsent_lexs, predicates = self.train_data
        print("Start training!")

        for epoch in range(self.EPOCHS):
            print('Start of epoch:', epoch+1, '/', self.EPOCHS)

            # Iterates over the whole dataset by splitting it in chunks of size BATCH_SIZE
            for i in tqdm(range(int(len(encsent_wids) / self.BATCH_SIZE))):

                # Training data gets split into batches
                batch_x, batch_x_senses, batch_y, batch_post_y, batch_lex_y, sent_length, (inds, onehot_pred) = \
                    generate_batches(encsent_wids, encsent_sids, encsent_roles, encsent_posts, encsent_lexs, predicates, self.BATCH_SIZE, seq=i)

                # Feeds the placeholders of the computational graph and runs the training
                session.run([self.train], feed_dict={self.word_ids: batch_x,
                                                     self.roles_ids: batch_y,
                                                     self.senses_ids: batch_x_senses,
                                                     self.post_ids:  batch_post_y,
                                                     self.lex_ids:   batch_lex_y,
                                                     self.sequence_lengths: sent_length,
                                                     self.predicates_ids:   inds,
                                                     self.onehot_preds:     onehot_pred})
            self.evaluation(session)

            # Save the model of this epoch to disk
            path = self.saver.save(session, MODEL_DIRECTORY + get_date())
            print("Model saved in path: %s" % path)


    def evaluation(self, session):
        """
        It is responsible for evaluating the performance of the classifier given a session.
        It prints the scores of the metrics.
        """

        # Unpack development data: encoded words, senses, roles, post tags, lexical tags and the indicies of the predicates for each sentence.
        encsent_wids, encsent_sids, encsent_roles, encsent_posts, encsent_lexs, predicates = self.dev_data
        counters = {'TP': 0, 'FP': 0, 'FN': 0, 'TOT': 0}  # Counters for the metrics
        prediction_truth = []

        print("Start evaluation!")

        # Iterates over the whole dataset
        for q in range(len(encsent_wids)):

            # Development data gets split into batches
            batch_x, batch_x_senses, batch_y, batch_post_y, batch_lex_y, sent_length, (inds, onehot_pred) = \
                generate_batches(encsent_wids, encsent_sids, encsent_roles, encsent_posts, encsent_lexs, predicates, 1, seq=q)

            # Feeds the placeholders of the computational graph and runs the evaluation
            preds = session.run([self.predictions_probabilities], feed_dict={self.word_ids:  batch_x,
                                                                             self.roles_ids: batch_y,
                                                                             self.senses_ids: batch_x_senses,
                                                                             self.post_ids:  batch_post_y,
                                                                             self.lex_ids:   batch_lex_y,
                                                                             self.predicates_ids:   inds,
                                                                             self.sequence_lengths: sent_length,
                                                                             self.onehot_preds:     onehot_pred})

            for i, truth in enumerate(encsent_roles[q]):
                # from preds: index 1 = sentence, index 2 = word, index 3 = role probability
                prediction = np.argmax(preds[0][0][i])
                #print(preds[0][0][i])

                prediction_truth.append((prediction, truth))

                counters['TOT'] += 1

                # TP: the prediction of the role was correct
                # FP: the word wasn't an argument but it was labeled with a role, or the role was wrong
                # FN: the word was an argument it was labeled null
                if prediction == truth and not prediction == 0:
                    counters['TP'] += 1
                elif not prediction == truth and not prediction == 0:
                    counters['FP'] += 1
                elif not prediction == truth and prediction == 0:
                    counters['FN'] += 1

        precision = counters['TP'] / (counters['TP'] + counters['FP'])
        recall = counters['TP'] / (counters['TP'] + counters['FN'])
        F1 = 2 * precision * recall / (precision + recall)

        print('The counters:', counters)
        print('Precision:', precision)
        print('Recall:', recall)
        print('F1:', F1)


    def test(self, session):
        """
        It is responsible for writing the output of the predictions over the test split of the dataset.
        """

        # Unpack development data: encoded words, senses, roles, post tags, lexical tags and the indicies of the predicates for each sentence.
        encsent_wids, encsent_sids, encsent_roles, encsent_posts, encsent_lexs, predicates = self.test_data
        id_role_dict = reverse_dict(self.dataset_test[1])
        sentences_wp = []
        
        print("Start testing!")
        
        # Iterates over the whole dataset
        for q in range(len(encsent_wids)):
            sentence = []
            
            # Test data gets split into batches
            batch_x, batch_x_senses, batch_y, batch_post_y, batch_lex_y, sent_length, (inds, onehot_pred) = \
                generate_batches(encsent_wids, encsent_sids, encsent_roles, encsent_posts, encsent_lexs, predicates, 1, seq=q)

            # Feeds the placeholders of the computational graph and runs the test
            preds = session.run([self.predictions_probabilities], feed_dict={self.word_ids:  batch_x,
                                                                             self.roles_ids: batch_y,
                                                                             self.senses_ids: batch_x_senses,
                                                                             self.post_ids:  batch_post_y,
                                                                             self.lex_ids:   batch_lex_y,
                                                                             self.predicates_ids:   inds,
                                                                             self.sequence_lengths: sent_length,
                                                                             self.onehot_preds:     onehot_pred})
            
            # Store all the predictions in the list 'sentences_wp'
            for i in range(len(encsent_wids[q])):
                prediction = np.argmax(preds[0][0][i])
                role = id_role_dict[prediction]
                sentence.append(role)
            sentences_wp.append(sentence)
        
        # Write all the predicitons to a file in the requested format
        wp = -1
        maxv = -1
        out_file = ""
        for sentence in self.dataset_test[0].sentences:
            if sentence.n_predicates == 0:
                for word in sentence.words:
                    out_file += word.raw + "\t_\n"
            else:
                for iw, word in enumerate(sentence.words):
                    roles_cols = ''
                    for pp in range(sentence.n_predicates):
                        roles = sentences_wp[wp+pp+1]
                        maxv = max(wp+pp+1, maxv)
                        role = roles[iw] if not roles[iw] == 'NULL' else '_'
                        roles_cols += "\t" + role
                    out_file += word.raw + roles_cols + '\n'
                wp = maxv
            out_file += '\n'

        with open('out.txt', 'w') as f:
            f.write(out_file)


    def execute_model(self):
        self.build_graph()  # Build the TensorGlow Graph of this neural architectur

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #self.saver.restore(sess, "mods_mod1_out/20180912063145")

            self.training(sess)
            self.evaluation(sess)
            self.test(sess)
