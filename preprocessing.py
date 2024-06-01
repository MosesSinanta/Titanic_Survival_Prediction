from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(data):
    data = data.drop(columns = ["PassengerId", "Name", "Ticket", "Cabin"])

    X = data.drop(columns = ["Survived"])
    y = data["Survived"]

    numerical_features = ["Age", "Fare", "SibSp", "Parch"]
    numerical_transformer = Pipeline(steps = [
        ("imputer", SimpleImputer(strategy = "median")),
        ("scaler", StandardScaler())
    ])

    categorical_features = ["Pclass", "Sex", "Embarked"]
    categorical_transformer = Pipeline(steps = [
        ("imputer", SimpleImputer(strategy = "most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown = "ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers = [
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ])

    X_preprocessed = preprocessor.fit_transform(X)

    return X_preprocessed, y
