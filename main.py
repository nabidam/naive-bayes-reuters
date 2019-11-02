import numpy as np
from Reuters import Reuters
from NaiveBayes import NaiveBayes

reuters = Reuters("data/")
train_documents, train_labels, test_documents, test_labels, label2topic = reuters.read()

train_documents = np.asarray(train_documents)
train_labels = np.asarray(train_labels)
test_documents = np.asarray(test_documents)
test_labels = np.asarray(test_labels)

print(train_labels, label2topic)

nb = NaiveBayes(np.unique(train_labels))
print("---------------- Training In Progress --------------------")

nb.train(train_documents, train_labels)
print('----------------- Training Completed ---------------------')

pclasses = nb.test(test_documents)
print('----------------- Test Completed ---------------------')

tps = np.zeros((len(label2topic),))
fps = np.zeros((len(label2topic),))
fns = np.zeros((len(label2topic),))
tns = np.zeros((len(label2topic),))
number_of_docs = np.zeros((len(label2topic),))
correct_predicted = np.zeros((len(label2topic),))

for idx, c in enumerate(pclasses):
    number_of_docs[c] += 1
    if c == test_labels[idx]:
        tps[test_labels[idx]] += 1
        correct_predicted[c] += 1
    else:
        fns[test_labels[idx]] += 1
        fps[c] += 1
tns = len(test_documents) - tps - fns - fps


test_acc = np.sum(pclasses == test_labels)/float(test_labels.shape[0])
each_accuracies = (tps + tns)/(tps + tns + fns + fps)

each_precisions = tps/(tps+fps)

each_recalls = tps/(tps+fns)

overal_accuracies = 0
print("Test results:")
for l in label2topic:
    print("Topic: ", l[0])
    if number_of_docs[l[1]] == 0:
        print("No TEST Document belongs to this topic...")
        print("###############################################")
        overal_accuracies += each_accuracies[l[1]] * (tps[l[1]] + fns[l[1]])
    else:
        print("--> accuracy: ", each_accuracies[l[1]], "%")
        overal_accuracies += each_accuracies[l[1]] * (tps[l[1]] + fns[l[1]])
        print("--> precision: ", each_precisions[l[1]])
        print("--> recall: ", each_recalls[l[1]])
        print("--> f1-measure: ", 2 * (each_precisions[l[1]] * each_recalls[l[1]])/(
            each_precisions[l[1]] + each_recalls[l[1]]))
        print("###############################################")

each_precisions = each_precisions[~np.isnan(each_precisions)]
each_recalls = each_recalls[~np.isnan(each_recalls)]

precision = np.mean(each_precisions)
recall = np.mean(each_recalls)

f1 = 2*(precision * recall)/(precision + recall)

print("Test Set Examples: ", test_labels.shape[0])

# print("Test Set Accuracy: ", test_acc*100, "%")
print("Test Set Accuracy: ", (overal_accuracies/len(test_documents))*100, "%")
print("Test Set Precision: ", precision)
print("Test Set recall: ", recall)
print("Test Set f1-measure: ", f1)
