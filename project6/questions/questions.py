import nltk
import sys
import os
import string
import math

FILE_MATCHES = 3
SENTENCE_MATCHES = 10


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    map_dict = dict()

    with os.scandir(directory) as files:
        for file in files:
            if file.name.endswith(".txt") and file.is_file():
                f = open(file.path, mode='r')
                all_lines = f.read()
                map_dict[file.name] = all_lines
    
    return map_dict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    all_words = [
        word.lower() for word in nltk.word_tokenize(document)
        if not word in string.punctuation and not word in nltk.corpus.stopwords.words("english")
    ]

    all_words.sort()

    return all_words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = set()
    for filename in documents:
        words.update(documents[filename])

    idfs_values = dict()
    for word in words:
        f = sum(word in documents[filename] for filename in documents.keys())
        idf = math.log(len(documents) / f)
        idfs_values[word] = idf

    return idfs_values


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idf = {filename:0 for filename in files}

    for filename in files:
        list_of_words = files[filename]
        result = 0.0
        for q_word in query:
            tf = list_of_words.count(q_word)
            idf = idfs[q_word] if q_word in list_of_words else 0
            result = (tf * idf) + result
        tf_idf[filename] = result

    tf_idf_list_sorted = sorted(tf_idf, key=tf_idf.get, reverse=True)

    return tf_idf_list_sorted[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    dict_result = {
        sentence: {
            "idf": 0.0,
            "density": 0.0
        } for sentence in sentences
    }

    for sentence in sentences:
        idf = 0.0
        total_words_found = 0
        for q_word in query:
            if q_word in sentences[sentence]:
                total_words_found += 1
                idf += idfs[q_word]
        
        density = float(total_words_found) / len(sentences[sentence])
        
        dict_result[sentence]["idf"] = idf
        dict_result[sentence]["density"] = density
    
    n_top_sentences = sorted(dict_result.items(), key=lambda x: (x[1]["idf"], x[1]["density"]), reverse=True)

    return [x[0] for x in n_top_sentences[:n]]


if __name__ == "__main__":
    main()
