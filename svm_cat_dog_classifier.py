import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

digits = datasets.load_digits()

X = digits.data
y = digits.target

y = (y > 4).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear')

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nActual Labels:")
print(y_test[:10])

print("\nPredicted Labels:")
print(pred[:10])

print("\nAccuracy:", accuracy_score(y_test, pred))