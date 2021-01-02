# Natural Language Processing Homework 3

Read the report for the details.

**Keywords: words embeddings, semantic disambiguation, semantic roles labeling**

_I suggest to read the project files in this order:_

1. [run.py](https://github.com/matteoprata/m-natural-language-processing-hw3/blob/master/src/run.py) - It is the access point of the project, a straightforward interface to easily prepare all the input data for the Neual Architecture by leveraging external modules and to execute the Neural Architecture of choice. I would read this module first to have a clear overview of the whole project, after this, I would move to the submodules required by run.py.

2. [preprocessing_dataset.py](https://github.com/matteoprata/m-natural-language-processing-hw3/blob/master/src/preprocessing_dataset.py) - It is a module containing three classes responsible for defining the structure of the parsed dataset. The three classes are the following: Dataset, Sentence and Word. Intuitively the Dataset is a set of sentences and a sentence is a sequence of words.

3. [preprocessing_parser.py](https://github.com/matteoprata/m-natural-language-processing-hw3/blob/master/src/preprocessing_parser.py) - It is a module containing the functions to parse the input dataset. The access point is the function *parse_dataset*, for which given the directory of a dataset, it parses it and it produces useful dictionaries for the encoding of the dataset itself. It also disambiguates the dataset by reading the disambiguations from an input file.

4. [preprocessing_vocabularies.py](https://github.com/matteoprata/m-natural-language-processing-hw3/blob/master/src/preprocessing_vocabularies.py) - It is a module containing the functions to build the lemma embeddings matrix and the sense embeddings matrix. Both functions also return dictionaries that map a key (lemma or sense) to the id of the vector in the embeddings matrix.

5. [preprocessing_embs_loader.py](https://github.com/matteoprata/m-natural-language-processing-hw3/blob/master/src/preprocessing_embs_loader.py) - It is a module containing the functions to read the embeddings matricies stored on disk. 

6. [preprocessing_dataset_enc.py](https://github.com/matteoprata/m-natural-language-processing-hw3/blob/master/src/preprocessing_dataset_enc.py) - It is a module containing the functions to encode the dataset according to the ids in the input dictionaries and to generate mini batches.

7. [neural_architecture.py](https://github.com/matteoprata/m-natural-language-processing-hw3/blob/master/src/neural_architecture.py) - It is the module containing the class NeuralArchitecture, which is the skeleton of the three implemented models. Through this, by calling the function *execute_model* a TensorFlow graph evaluated. In particular by calling the functions *train*, *evaluation* and *test* the proper operations of the graph get executed.

8. [model1.py](https://github.com/matteoprata/m-natural-language-processing-hw3/blob/master/src/model1.py) - It is a module containing the function to build the TensorFlow graph of model 1 (see report).


9. [model2.py](https://github.com/matteoprata/m-natural-language-processing-hw3/blob/master/src/model2.py) - It is a module containing the function to build the TensorFlow graph of model 2 (see report).

10. [model3.py](https://github.com/matteoprata/m-natural-language-processing-hw3/blob/master/src/model3.py) - It is a module containing the function to build the TensorFlow graph of model 3 (see report).

**others...** 
*utilities.py* and *constants.py* contain just irrelevant information to understand the system.


