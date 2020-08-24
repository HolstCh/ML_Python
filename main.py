from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


breast_cancer_data = load_breast_cancer()
#load data into variable

print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)
 #create 4 training/test variables with 80/20 split; each data portion has its own labels(the output of the data portion)

print(len(training_data))
print(len(training_labels))

accuracies = []

for k in range(1,101):
  classifier = KNeighborsClassifier(n_neighbors=k)
# create object with k value of 3 (k is number of points being evaluated near uknown point in question)
  classifier.fit(training_data, training_labels)
#training 80 percent of entire dataset
  validation_score = classifier.score(validation_data, validation_labels)
  accuracies.append(validation_score)
# using score fxn to see how accurate the training is by testing the set that has NOT been used yet for non bias outcome; the 20 percent split data portion. this proves that the chosen k value/n_neighbors = 3 is quite accurate; However, can find a better k by using for loop and starting at k=1:
# for k in range(1,101):
# create classifier (n_neighbor = k); WAS n_neighbor = 3, with valid score 0.94...
# train classifier
# report validation score
k_list = [*range(1,101,1)]
# x-axis of plot is k values
# y-axis is accuracies of each k value


plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()