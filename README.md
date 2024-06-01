# Titanic Survival Prediction

This project conducts a comparative analysis of multiple machine learning algorithms applied to the Titanic dataset to predict passenger survival.

#### Disclaimer
This project is not the first of its kind, and there are many similar projects on the same topic. It is created solely for educational purposes.



## Deployment

#### 1. Change the model type you want to run by commenting 4 lines in line 21 to 25 in main.py
```
#model = LogisticRegression(random_state = 42)
#model = SVC(kernel = "sigmoid", random_state = 42)
#model = DecisionTreeClassifier(random_state = 42)
#model = RandomForestClassifier(n_estimators = 190, random_state = 42)
model = KNeighborsClassifier()
```

#### 2. Change the JSON file output name in line 54 in main.py (optional)
```
with open("K-Nearest Neighbor results.json", "w") as file:
```

#### 3. Run main.py in terminal
```
python main.py
```



## Results

Past-run results can be seen in ```Logistic Regression results.json```, ```Support Vector Machines results.json```, ```Decision Tree results.json```, ```Random Forest results.json``` and ```K-Nearest Neighbors results.json```

This project also includes documentation in PDF.



## Authors

#### [MosesSinanta](https://github.com/MosesSinanta/)
A random nerd who likes doing fun projects.
