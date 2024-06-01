import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from preprocessing import preprocess_data

data = pd.read_csv("titanic.csv")

X, y = preprocess_data(data)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)

#model = LogisticRegression(random_state = 42)
#model = SVC(kernel = "sigmoid", random_state = 42)
#model = DecisionTreeClassifier(random_state = 42)
#model = RandomForestClassifier(n_estimators = 190, random_state = 42)
model = KNeighborsClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
accuracy = "{:.3f}".format(accuracy)

conf_matrix = confusion_matrix(y_val, y_pred)

results = {
    "accuracy": accuracy,
    "confusion_matrix": conf_matrix.tolist()
}

with open("K-Nearest Neighbor results.json", "w") as file:
    json.dump(results, file)

print("Results saved to JSON")
