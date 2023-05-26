# Multinomial-Naive-Bayes-from-Scratch

## Description

This repository contains a Jupyter notebook implementing the Multinomial Naive Bayes algorithm from scratch for an email classification task of SPAM or HAM. The notebook also includes a comparison of the results obtained with the scikit-learn implementation of Multinomial Naive Bayes.

If you are unable to render the notebook on github, then please use this [Notebook Link](https://nbviewer.org/github/akash18tripathi/Multinomial-Naive-Bayes-from-Scratch/blob/main/Multinomial-Naive-Bayes-from-Scratch.ipynb) .

## What notebook has to offer?

By following along with the notebook, you will gain a deeper understanding of Multinomial Naive Bayes and its application in Classification tasks.


## Dataset

The dataset is about Spam SMS. There is 1 attribute that is the message, and the class label which could be spam or ham. The data is present in spam.csv. It contains about 5-6000 samples. For your convinience the data is already pre-processed and loaded, but I suggest you to just take a look at the code for your own knowledge, and parts vectorization is left up to you which could be easily done.

## Multinomial Naive Bayes for SPAM/HAM Email Classification

Multinomial Naive Bayes is a probabilistic classification algorithm commonly used for text classification tasks, such as classifying emails as SPAM or HAM. It leverages the principles of Bayes' theorem to calculate the probability of an email belonging to a particular class given its features.

### Text Preprocessing

Before applying the Multinomial Naive Bayes algorithm, the text data needs to be preprocessed. The provided `text_process` function performs the following steps:

1. **Remove Punctuation**: All punctuation marks are removed from the text using the `string.punctuation` library.

2. **Remove Stopwords**: Stopwords are common words that do not carry significant meaning and are often removed from text data. The function removes stopwords using the `stopwords.words('english')` list, which includes common English stopwords. Additional stopwords such as 'u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure' are also included.

3. **Return Cleaned Text**: The processed text is returned as a list of cleaned words, joined together by a space.

### Prior Calculation

In Multinomial Naive Bayes, prior probabilities are calculated for each class (SPAM or HAM) based on the training data. These probabilities represent the likelihood of an email being classified as a specific class without considering any features.

The formula to calculate the prior probability of a class is:

`P(class) = count(class) / count(total)`


where:
- `count(class)` is the number of emails in the training set belonging to the class.
- `count(total)` is the total number of emails in the training set.

The prior probability provides a baseline probability for each class and is used as a starting point for calculating the posterior probability during prediction.

### CountVectorizer for Frequency Features

CountVectorizer is a feature extraction technique used in text processing to convert text data into numerical feature vectors. It represents the frequency of each word in the text corpus.

In the context of Multinomial Naive Bayes, CountVectorizer is used to transform the preprocessed text data into a matrix of word frequencies. Each row of the matrix represents an email, and each column represents a unique word in the training dataset. The matrix elements denote the frequency of each word in the corresponding email.

CountVectorizer is a common feature extraction step in text classification tasks as it enables the algorithm to work with numerical data rather than raw text.

Please note that the provided code snippet defines the `text_process` function for text preprocessing, but the actual usage and integration with Multinomial Naive Bayes is not shown.

The Jupyter Notebook in this repository demonstrates each step with detailed explanations and code snippets.

### Smoothing Priors in Multinomial Naive Bayes

Smoothing priors, also known as additive smoothing or Laplace smoothing, are used in Multinomial Naive Bayes to address the issue of zero probabilities. When a word in the test data is not present in the training data, the conditional probability of that word given a class becomes zero, leading to a probability of zero for that class. This can result in inaccurate predictions and instability in the algorithm.

To overcome this problem, smoothing priors are introduced. The most commonly used smoothing technique is called Laplace smoothing, which adds a small constant value (typically denoted as alpha) to the word counts for each class during training. By doing so, the probability of an unseen word becomes non-zero and ensures that no probability is zero for any class or feature.

The formula to calculate the smoothed probability of a word given a class is as follows:

`P(word|class) = (count(word, class) + alpha) / (count(class) + alpha * V)`


where:
- `count(word, class)` is the number of occurrences of the word in the training examples of the given class.
- `count(class)` is the total count of all words in the training examples of the given class.
- `alpha` is the smoothing parameter, typically set to a small value.
- `V` is the total number of unique words in the training dataset.

The addition of alpha in the numerator and alpha multiplied by the total number of unique words in the denominator ensures that no probability is zero and prevents the algorithm from assigning zero probabilities to unseen words.

Smoothing priors play a crucial role in improving the stability and performance of Multinomial Naive Bayes, especially in scenarios where there are unseen words in the test data that were not present in the training data.


### Contribution

Any Contributions are most welcome. Feel free to play around the notebook and suggest changes.


