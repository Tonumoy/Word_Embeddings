# Word Embeddings
Word embedding is a technique where individual words of a domain or language are represented as real-valued vectors in a lower dimensional space. Word embeddings are considered to be one of the successful applications of unsupervised learning at present. They do not require any annotated corpora. Embeddings use a lower-dimensional space while preserving semantic relationships.
Some popular word embedding methods to extract features from text are:

1. **Bag of words** - Bag of words is a simple and popular technique for feature extraction from text. Bag of word model processes the text to find how many times each word appeared in the sentence. This is also called as vectorization.  

2. **TF-IDF** - TF-IDF is a popular word embedding technique for extracting features from corpus or vocabulary. This is a statistical method to find how important a word is to a document all over other documents.   

    The full form of TF is Term Frequency. In TF, we are giving some scoring for each word or token based on the frequency of that word. The frequency of a word is dependent on the length of the document. Means in large size of document a word occurs more than a small or medium size of the documents. So to overcome this problem we will divide the frequency of a word with the length of the document (total number of words) to normalize.By using this technique also, we are creating a sparse matrix with frequency of every word.
    
    Formula to calculate Term Frequency (TF)
    
    TF = no. of times term occurrences in a document / total number of words in a document  
    
    The full form of IDF is Inverse Document Frequency. Here also we are assigning  a score value  to a word , this scoring  
    value explains how a word is rare across all documents. Rarer words have more IDF score.

    Formula to calculate Inverse Document Frequency (IDF) :-  
    IDF = log base e (total number of documents / number of documents which are having term )  
    Formula to calculate complete TF-IDF value is:

    TF-IDF  = TF * IDF  
    
    TF-IDF value will be increased based on frequency of the word in a document. Like Bag of Words in this technique also we  
    cannot get any semantic meaning for words.This technique is mostly used for document classification and also successfully  
    used by search engines like Google, as a ranking factor for content. 


3. **Word2vec** - Word2vec is an algorithm invented at Google for training word embeddings. word2vec relies on the distributional hypothesis. The distributional hypothesis states that words which, often have the same neighboring words tend to be semantically similar. This helps to map semantically similar words to geometrically close embedding vectors.  

4. **Fastext** - FastText is an open-source, free library from Facebook AI Research(FAIR) for learning word embeddings and word classifications. This model allows creating unsupervised learning or supervised learning algorithm for obtaining vector representations for words. It also evaluates these models. FastText supports both CBOW and Skip-gram models (*Continuous Bag of Words Model (CBOW) and Skip-gram Both are architectures to learn the underlying word representations for each word by using neural networks*).

5. **ELMO (Embeddings for Language models)** - Embeddings from Language Models (ELMo) is also a powerful computational model that converts words into numbers. This vital process allows machine learning models (which take in numbers, not words, as inputs) to be trained on textual data. It achieved state-of-the-art performance on many popular tasks including question-answering, sentiment analysis, and named-entity extraction. ELMo can uniquely account for a word’s context. Previous language models such as GloVe, Bag of Words, and Word2Vec simply produce an embedding based on the literal spelling of a word. They do not factor in how the word is being used.   
