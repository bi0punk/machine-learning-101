import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split


def load_data(train_path="titanic/train.csv", test_path="titanic/test.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def preprocess(df):
    df = df.drop(["PassengerId", "Ticket", "Cabin"], axis=1, errors="ignore")
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True, dummy_na=False)
    return df


def engineer_features(train, test):
    for df in [train, test]:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.")
        df["Title"] = df["Title"].replace(
            ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"],
            "Rare"
        )
        df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
        df["Title"] = df["Title"].replace("Mme", "Mrs")

    title_mapping = {t: i for i, t in enumerate(train["Title"].unique())}
    for df in [train, test]:
        df["Title"] = df["Title"].map(title_mapping).fillna(-1)

    return train, test


def split_features_target(train):
    X = train.drop(["Survived", "Name"], axis=1)
    y = train["Survived"]
    return X, y


def prepare_test(test, X_train_columns):
    test = test.drop(["Name"], axis=1, errors="ignore")
    for col in X_train_columns:
        if col not in test.columns:
            test[col] = 0
    return test[X_train_columns]


def main():
    print("=" * 60)
    print("Titanic - Pipeline de Machine Learning")
    print("=" * 60)

    train, test = load_data()
    print(f"\nTrain shape: {train.shape}, Test shape: {test.shape}")

    train = preprocess(train)
    test_id = test["PassengerId"]
    test = preprocess(test)

    train, test = engineer_features(train, test)

    X, y = split_features_target(train)
    X_test = prepare_test(test, X.columns)

    num_cols = X.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy="median")
    X[num_cols] = imputer.fit_transform(X[num_cols])
    X_test[num_cols] = imputer.transform(X_test[num_cols])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nEntrenando Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print("\n=== Resultados ===")
    print(f"Precisión (validation): {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=["Fallecido", "Sobreviviente"]))

    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    test_pred = model.predict(X_test)
    submission = pd.DataFrame({
        "PassengerId": test_id,
        "Survived": test_pred
    })
    submission.to_csv("titanic/predictions.csv", index=False)
    print("\nPredicciones guardadas en titanic/predictions.csv")
    print(f"Distribución: {pd.Series(test_pred).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
