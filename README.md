# naive-bayes-reuters
Classifying Reuters news articles using Naive Bayes Classifier.

# Reuters
Class Reuters gets a location argument that specifies *.sgm files of dataset, then you can use `read()` function to read files into 5 output lists corresponding: 
train_documents, train_labels, test_documents, test_labels, label2topic (list that maps labels to topics string).

# NaiveBayes
Class NaiveBayes is used for initializing the classifier. then you train classifier with train data, finally you can test the trained model using `test()` function.
This classifier first preprocess data by removing punctuation marks and numbers, removing extra white spaces, and making all letters lower form.
Then tokenize the documents for training the model.
