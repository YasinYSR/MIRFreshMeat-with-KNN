from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import csv

with open("MIRFreshMeats.csv", 'r') as x:
    dataset = list(csv.reader(x, delimiter=","))
dataset = np.array(dataset)

wavenumbers = dataset[:,0]
dataset = np.delete(dataset, 0, axis=0)

sample = list(np.delete(dataset, 0, axis=1))
sample = np.array(sample, dtype=float)
sample_t = sample.transpose()

t = np.arange(1,121,1)
t[:40] = 1
t[40:80] = 2
t[80:120] =3


####################################################################################
X_train, X_test, y_train, y_test = train_test_split(sample_t, t, train_size= 0.25)

neighbors = np.arange(1, 31)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
  
# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto', metric='euclidean')
    knn.fit(X_train, y_train)
      
    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
  
# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
  
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.grid()
plt.show()