# Import general libraries
import numpy as np
import pandas as pd
# Import libraries for preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
# Import libraries for classification
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, accuracy_score, cohen_kappa_score

def clean_data(df, features, target=None):
    """
    If the target column name is provided, the function will treat it as training
    data, otherwise as test data.
    Rows with missing features or target labels are removed.
    Target labels are imputed by backward filling.
    """
    # Remove trailing whitespace from column names
    df.columns = df.columns.str.strip()
    # Remove empty columns
    df.dropna(how='all', axis='columns', inplace=True)
    if target is not None:
        # Impute values for missing labels by backward filling
        for ith_row in range(len(df)-2, -1, -1):
            if pd.isnull(df.iloc[ith_row, 0]):
                df.loc[ith_row, target] = df.loc[ith_row+1, target]
        # Remove rows with missing labels
        df.dropna(subset=[target] + features, inplace=True)
        # Remove rows with the stand deviation of the measurements
        df = df[~df[target].str.contains("std|stddev|std dev|standard deviation", case=False)]
    else:
        df.dropna(subset=features, inplace=True)
    if ("TAS" in features) & ("Na2O" in features) & ("K2O" in features):
        df.TAS = df.Na2O + df.K2O
    return df

def model_validation(train_file, target, features):
    # Load training data
    train = load_clean("../Data/Preprocessed/" + train_file, features, target)
    train.loc[:, target] = train.apply(lambda row : row.Source + "_mafic" if row.SiO2 < 55 else row.Source, axis=1) 
    X = np.array(train[features])
    y = np.array(train[target])
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    ################################################################
    # Load models for validation
    # Perform a grid search for hyperparameters and cross-validation
    # to determine the best model.
    ###############################################################

    # Define the pipelines for each model
    nn_pipe = Pipeline([("scaler", StandardScaler()), ("nn", MLPClassifier(max_iter=2000))])

    # Define grid of hyperparameters for search
    knn_grid = {"n_neighbors" : range(1, 16, 2), 
                "weights" : ("uniform", "distance")}
    rf_grid =  {"n_estimators" : range(1, 51, 5), 
                "max_features" : (None, "sqrt", "log2"), 
                "max_depth" : [None] + list(range(2, 21, 2))}
    nn_grid =  {"nn__hidden_layer_sizes" : [(7,), (10,), (15,), (25,), (50,), (5,5), (10, 10), (25, 25)],
                "nn__activation" : ("relu", "logistic"),
                "nn__alpha" : [0.001, 0.01, 0.1]}

    cv = StratifiedKFold(n_splits=3, random_state=42)
    score = {
            "kappa": make_scorer(cohen_kappa_score),
            "accuracy": make_scorer(accuracy_score),
    }
    
    # Define the grid search
    gs_knn = GridSearchCV(KNeighborsClassifier(), knn_grid, cv=cv, scoring=score, refit="kappa")
    gs_rf = GridSearchCV(RandomForestClassifier(), rf_grid, cv=cv, scoring=score, refit="kappa")
    gs_nn = GridSearchCV(nn_pipe, nn_grid, cv=cv, scoring=score, refit="kappa")
    
    # Perform the grid search
    gs_knn.fit(X_train, y_train)
    gs_rf.fit(X_train, y_train)
    gs_nn.fit(X_train, y_train)

    # Validate the models
    y_knn = gs_knn.best_estimator_.predict(X_val)
    y_nn = gs_nn.best_estimator_.predict(X_val)
    y_rf = gs_rf.best_estimator_.predict(X_val)
    nn_probs = gs_nn.best_estimator_.predict_proba(X_val)
    rf_probs = gs_rf.best_estimator_.predict_proba(X_val)
    avg_probs = (rf_probs + nn_probs) / 2
    y_avg = label_encoder.inverse_transform(np.argmax(avg_probs, axis=1))

    # Export results to a file
    models = ["TREE", "KNN", "KNN_norm","SVM","NN","RF", "AVG"]
    preds = [y_knn, y_nn, y_rf, y_avg]
    accuracies = []
    kappas = []
    for pred in preds:
        accuracies.append(accuracy_score(y_val, pred))
        kappas.append(cohen_kappa_score(y_val, pred))

    # Create a data frame with the scores and save to a file
    df = pd.DataFrame({"Model": models,
                       "Accuracy": accuracies,
                       "Kappa": kappas})
    return df

def model_selection(train_file, target, features):
    # Load training data
    train = load_clean("../Data/Preprocessed/" + train_file, features, target)
    train.loc[:, target] = train.apply(lambda row : row.Source + "_mafic" if row.SiO2 < 55 else row.Source, axis=1) 
    X = np.array(train[features])
    y = np.array(train[target])

    ################################################################
    # Load models for validation
    # Perform a grid search for hyperparameters and cross-validation
    # to determine the best model.
    ###############################################################

    # Load the grid search objects from model_validation
    gs_knn =        load("../Output/Models/gs_KNNnorm_" + train_file[0:-4] + ".joblib")
    gs_knn_nonorm = load("../Output/Models/gs_KNN_" + train_file[0:-4] + ".joblib")
    gs_rf =         load("../Output/Models/gs_RF_" + train_file[0:-4] + ".joblib")
    gs_tree =       load("../Output/Models/gs_TREE_" + train_file[0:-4] + ".joblib")
    gs_svc =        load("../Output/Models/gs_SVC_" + train_file[0:-4] + ".joblib")
    gs_nn =         load("../Output/Models/gs_NN_" + train_file[0:-4] + ".joblib")
    
    # Perform the grid search
    gs_knn.fit(X, y)
    gs_knn_nonorm.fit(X, y)
    gs_rf.fit(X, y)
    gs_tree.fit(X, y)
    gs_svc.fit(X, y)
    gs_nn.fit(X, y)

    # Print the best score for each model
    print("Best scores:")
    print("KNN: {}".format(gs_knn.best_score_))
    print("KNN (No Normalization): {}".format(gs_knn_nonorm.best_score_))
    print("Random Forest: {}".format(gs_rf.best_score_))
    print("Decision Tree: {}".format(gs_tree.best_score_))
    print("SVM: {}".format(gs_svc.best_score_))
    print("Neural Network: {}".format(gs_nn.best_score_))

    # Save models
    dump(gs_knn_nonorm.best_estimator_, "../Output/Models/KNNnorm_" + train_file[0:-4] + ".joblib")
    dump(gs_knn.best_estimator_, "../Output/Models/KNN_" + train_file[0:-4] + ".joblib")
    dump(gs_rf.best_estimator_, "../Output/Models/RF_" + train_file[0:-4] + ".joblib")
    dump(gs_tree.best_estimator_, "../Output/Models/TREE_" + train_file[0:-4] + ".joblib")
    dump(gs_svc.best_estimator_, "../Output/Models/SVC_" + train_file[0:-4] + ".joblib")
    dump(gs_nn.best_estimator_, "../Output/Models/NN_" + train_file[0:-4] + ".joblib")

def classify(test_files, target, features, train_file, prefix="predictions"):
    if train_file[-4] == '.':
        train_file = train_file[0:-4]

    # Load the grid search objects from model_validation
    knn =        load("../Output/Models/KNNnorm_" + train_file + ".joblib")
    knn_nonorm = load("../Output/Models/KNN_" + train_file + ".joblib")
    rf =         load("../Output/Models/RF_" + train_file + ".joblib")
    tree =       load("../Output/Models/TREE_" + train_file + ".joblib")
    svc =        load("../Output/Models/SVC_" + train_file + ".joblib")
    nn =         load("../Output/Models/NN_" + train_file + ".joblib")

    if isinstance(test_files, str):
        test_files = [test_files]
    for test_file in test_files:
        # Load test data
        # test_tb = load_clean("../Data/Preprocessed/" + test_file, features)
        test_tb = pd.read_csv("../Data/Preprocessed/" + test_file, delimiter='\t')
        X = np.array(test_tb.loc[:, features])

        # Predict
        y_knn = knn.predict(X)
        y_knn_nonorm = knn_nonorm.predict(X)
        y_tree = tree.predict(X)
        y_svc = svc.predict(X)
        y_nn = nn.predict(X)
        y_rf = rf.predict(X)
        nn_probs = nn.predict_proba(X)
        rf_probs = rf.predict_proba(X)
        avg_probs = (rf_probs + nn_probs) / 2
        label_encoder = LabelEncoder()
        label_encoder.fit(rf.classes_)
        y_avg = label_encoder.inverse_transform(np.argmax(avg_probs, axis=1))

        # Add the predictions of each classifier to the test file
        test_tb.loc[:, "TREE"] = y_tree
        test_tb.loc[:, "KNN"] = y_knn
        test_tb.loc[:, "KNN Norm"] = y_knn_nonorm
        test_tb.loc[:, "SVM"] = y_svc
        test_tb.loc[:, "NN"] = y_nn
        test_tb.loc[:, "RF"] = y_rf
        test_tb.loc[:, "AVG"] = y_avg

        # Add the predicted probabilities to the the test file
        models = [('RF', rf_probs), ('NN', nn_probs), ('AVG', avg_probs)]
        labels = rf.classes_
        for name, probs in models:
            columns = [f"{label}_{name}" for label in labels]
            test_tb[columns] = probs

        # Save the file
        file_name = test_file[0: -4]
        test_tb.to_csv("../Output/Classification/" + prefix + "_" + file_name + ".csv", index=False)
