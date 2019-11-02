import os
import sys
import string
from bs4 import BeautifulSoup


class Reuters:
    def __init__(self, location):
        self.location = location
        self.labels = self.getLabels()

    def read(self):
        labels = self.labels
        train_documents = []
        train_labels = []
        test_documents = []
        test_labels = []
        files = [self.location + "reut2-%03d.sgm" % r for r in range(0, 22)]
        train_label_docs, test_label_docs = self.readFiles(files)
        label2topic = []
        idx = 0
        for label in labels:
            if len(train_label_docs[label]):
                for doc in train_label_docs[label]:
                    train_documents.append(doc)
                    train_labels.append(idx)
            if len(test_label_docs[label]):
                for doc in test_label_docs[label]:
                    test_documents.append(doc)
                    test_labels.append(idx)
            label2topic.append([label, idx])
            idx += 1

        return train_documents, train_labels, test_documents, test_labels, label2topic

    def generate_tree(self, text):
        return BeautifulSoup(text, "html.parser")

    def getLabels(self):
        topics = open(self.location +
                      "all-topics-strings.lc.txt", "r").readlines()
        topics = [t.strip() for t in topics]
        for i in range(0, len(topics)):
            topics[i] = topics[i].lower()
        return topics

    def readFiles(self, files):
        reuters = []
        labels = self.labels
        train_label_docs = {l: [] for l in labels}
        test_label_docs = {l: [] for l in labels}
        for file in files:
            data = open(file, 'rb')
            text = data.read()
            data.close()
            tree = self.generate_tree(text.lower())
            for reuter in tree.find_all("reuters"):
                # has_topic = reuter.get('topics') == 'yes' if 1 else 0
                document_topics = reuter.find('topics').find_all('d')
                if len(document_topics) > 0:
                    document_text = reuter.find('text')
                    title = ''
                    body = ''

                    if document_text.title != None:
                        title = document_text.title.text
                    if document_text.body != None:
                        body = document_text.body.text
                    document_text = title + ' ' + body
                    t = document_topics[0].text
                    if reuter.get('cgisplit') == "training-set":
                        train_label_docs[t].append(document_text)
                    else:
                        test_label_docs[t].append(document_text)
                    # for topic in document_topics:
                    #     t = topic.text
                    #     if reuter.get('cgisplit') == "training-set":
                    #         train_label_docs[t].append(document_text)
                    #     else:
                    #         test_label_docs[t].append(document_text)
            print("Finished extracting information from file:", file)
        return train_label_docs, test_label_docs
