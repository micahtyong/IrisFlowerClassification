# 0. Import tools
import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics

# 1. Collect Data
iris = datasets.load_iris()

# 2. Choose Model
# Simple Linear Classifier
classifier = skflow.TensorFlowLinearClassifier(n_classes=3)

# Deep Neural Network Classifier
# classifier = skflow.TensorFlowDNNClassifier(hidden_units = [10, 20, 10], n_classes=3)

# 3. Train the Model

# iris.data is our m by n matrix, X, which corresponds to X having
# m examples and n features

# iris.target is the class that we are trying to predict. This would correspond
# to the m by 1 vector, y, where each entry is the true value in our dataset

classifier.fit(iris.data, iris.target)

# 4. Test the Model
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
print("Accuracy: %f" % score)

