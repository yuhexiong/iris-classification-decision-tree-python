import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

iris = pd.read_csv("data/Iris.csv")
iris = iris.dropna()
print("iris.head()")
print(iris.head())

print("iris.shape", iris.shape)

print(iris["Species"].unique() )

# data distribution plot
fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()

fig = iris[iris.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title(" Petal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()

# correlation matrix
plt.figure(figsize=(10,6)) 
cmap = sns.color_palette("Blues", as_cmap=True)
sns.heatmap(iris.corr(), annot=True, cmap=cmap, annot_kws={'size': 16})
plt.show()

X = np.array(iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])                   # Input
Y = np.array(iris["Species"])  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 100)

# model
clf = DecisionTreeClassifier(criterion = "entropy",
                                max_depth = 5,
                                min_samples_leaf = 3,
                                random_state = 100)
clf.fit(X_train, Y_train)

y_pred = clf.predict(X_test)   

print ("Accuracy:", accuracy_score(Y_test, y_pred)*100)
print ("Report:",  classification_report(Y_test, y_pred))

# confusion matrix
cm = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris["Species"].unique(), yticklabels=iris["Species"].unique(), annot_kws={'size': 20})
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# tree plot
plt.figure(figsize=(12, 8))
plot_tree(clf)
plt.show()