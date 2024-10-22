import marimo

__generated_with = "0.8.9"
app = marimo.App(width="full", app_title="Tephra correlation")


@app.cell
def __():
    import io
    import os
    import ast
    import functions
    import marimo                as     mo
    import numpy                 as     np
    import pandas                as     pd
    from joblib                  import dump, load
    from sklearn.metrics         import make_scorer, accuracy_score, cohen_kappa_score
    from sklearn.ensemble        import RandomForestClassifier
    from sklearn.pipeline        import Pipeline
    from sklearn.ensemble        import RandomForestClassifier
    from sklearn.neighbors       import KNeighborsClassifier
    from sklearn.preprocessing   import LabelEncoder, StandardScaler
    from sklearn.neural_network  import MLPClassifier
    from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
    return (
        GridSearchCV,
        KNeighborsClassifier,
        LabelEncoder,
        MLPClassifier,
        Pipeline,
        RandomForestClassifier,
        StandardScaler,
        StratifiedKFold,
        accuracy_score,
        ast,
        cohen_kappa_score,
        dump,
        functions,
        io,
        load,
        make_scorer,
        mo,
        np,
        os,
        pd,
        train_test_split,
    )


@app.cell
def __(mo):
    f = mo.ui.file(filetypes=[".csv"], multiple=False)
    return f,


@app.cell
def __(mo):
    mo.md("""# Correlation of tephras using Machine Learning""")
    return


@app.cell
def __(mo):
    mo.md("""### **Setup**""")
    return


@app.cell
def __(f, mo):
    mo.md(f"""
        Upload the csv file wiht the training data (the correlated tephras).
        {f}
    """)
    return


@app.cell
def __(all_features, f, functions, io, mo, np, pd):
    if f.name() is not None:
        _df            = pd.read_csv(io.StringIO(f.contents().decode()))
        features       = list(set(_df.columns.str.strip()) & set(all_features))
        df_temp        = functions.clean_data(_df, features)
        _non_features  = list(set(df_temp.select_dtypes(exclude=np.number).columns.tolist()).difference(set(all_features)))
        sources_column = mo.ui.radio(options=_non_features, inline=True, label="**Groups column**: ")
    return df_temp, features, sources_column


@app.cell
def __(all_features, f, functions, io, mo, pd):
    if f.name() is not None:
        _df           = pd.read_csv(io.StringIO(f.contents().decode()))
        file_features = list(set(_df.columns.str.strip()) & set(all_features))
        choices       = mo.ui.array([mo.ui.checkbox(value=True, label=feat) for feat in file_features])
        train         = functions.clean_data(_df, file_features)
    return choices, file_features, train


@app.cell
def __(choices, f, mo):
    if f.name() is not None:
        _output = mo.md(f"""
            Choose which elements you would like to use in the correlation.  
            If there are elements in the table that are not showing below, check the file for typos.
            {mo.hstack(choices)}
        """)
    else:
        _output = None
    _output
    return


@app.cell
def __(f, mo, sources_column):
    if f.name() is not None:
        _output = mo.md(f"""
            Select the column of the file containing the sources.
            {sources_column}
        """)
    else:
        _output = None
    _output
    return


@app.cell
def __(f, mo, sources_column):
    if (f.name() is not None) and (sources_column.value is not None):
        _output = mo.md("""
            ### **Model validation**
        """)
    else:
        _output = None
    _output
    return


@app.cell
def __(f, mo, sources_column):
    if (f.name() is not None) and (sources_column.value is not None):
        validate_models = mo.ui.checkbox(value=False, label="Validate models")
        _output = mo.md(f"""
            You can validate the models? Takes considerably longer than training.  
            Do you want to validate the models?  
            {validate_models}
        """)
    else:
        validate_models = None
        _output = None
    _output
    return validate_models,


@app.cell
def __(
    GridSearchCV,
    KNeighborsClassifier,
    LabelEncoder,
    MLPClassifier,
    Pipeline,
    RandomForestClassifier,
    StandardScaler,
    StratifiedKFold,
    accuracy_score,
    cohen_kappa_score,
    make_scorer,
):
    from scipy.spatial.distance import jensenshannon
    label_encoder = LabelEncoder()
    # Define the pipelines for each model
    nn_pipe  = Pipeline([("scaler", StandardScaler()), ("nn", MLPClassifier(max_iter=2000))])

    # Define grid of hyperparameters for search
    knn_grid = {"n_neighbors"            : range(1, 16, 2), 
                "weights"                : ("uniform", "distance")}
    rf_grid  = {"n_estimators"           : range(1, 51, 3), 
                "max_features"           : (None, "sqrt", "log2"), 
                "max_depth"              : [None] + list(range(2, 21, 2))}
    nn_grid  = {"nn__hidden_layer_sizes" : [(7,), (10,), (15,), (25,), (50,), (5,5), (10, 10), (25, 25)],
                "nn__activation"         : ("relu", "logistic"),
                "nn__alpha"              : [0.001, 0.01, 0.1]}
    score    = {"kappa"                  : make_scorer(cohen_kappa_score),
                "accuracy"               : make_scorer(accuracy_score)}
    cv       = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Define the grid search
    gs = {}
    #gs["jsd"] = GridSearchCV(KNeighborsClassifier(metric=jensenshannon), knn_grid, cv=cv, scoring=score, refit="kappa")
    gs["knn"] = GridSearchCV(KNeighborsClassifier(metric='euclidean'),   knn_grid, cv=cv, scoring=score, refit="kappa")
    gs["rf"]  = GridSearchCV(RandomForestClassifier(),                   rf_grid,  cv=cv, scoring=score, refit="kappa")
    gs["nn"]  = GridSearchCV(nn_pipe,                                    nn_grid,  cv=cv, scoring=score, refit="kappa")
    return (
        cv,
        gs,
        jensenshannon,
        knn_grid,
        label_encoder,
        nn_grid,
        nn_pipe,
        rf_grid,
        score,
    )


@app.cell
def __():
    # Load training data
    from scipy.spatial import distance
    def dist(a,b):
        return distance.jensenshannon(a/100, b/100)
    return dist, distance


@app.cell
def __(
    accuracy_score,
    cohen_kappa_score,
    features,
    gs,
    label_encoder,
    np,
    pd,
    sources_column,
    train,
    train_test_split,
    validate_models,
):
    if (validate_models is not None) and (validate_models.value):
        # Load training data
        _X             = np.array(train[features])
        _y             = np.array(train[sources_column.value])
        label_encoder.fit(_y)
        _X_train, _X_val, _y_train, _y_val = train_test_split(_X, _y, test_size=0.2, stratify=_y, random_state=42)

        # Perform the grid search
        _ys = {}
        for _g in gs:
            gs[_g].fit(_X_train, _y_train)
            _ys[_g] = gs[_g].best_estimator_.predict(_X_val)

        # Validate the models
        _nn_probs  =  gs["nn"].best_estimator_.predict_proba(_X_val)
        _rf_probs  =  gs["rf"].best_estimator_.predict_proba(_X_val)
        _avg_probs = (_rf_probs + _nn_probs) / 2
        _ys["avg"] = label_encoder.inverse_transform(np.argmax(_avg_probs, axis=1))

        # Store results in a data frame
        _accuracies = []
        _kappas     = []
        for pred in _ys.values():
            _accuracies.append(accuracy_score(_y_val, pred))
            _kappas.append(cohen_kappa_score(_y_val, pred))
        _df = pd.DataFrame({"Model"   : _ys.keys(),
                           "Accuracy" : _accuracies,
                           "Kappa"    : _kappas})
    else:
        _df = None
    # Show results
    _df
    return pred,


@app.cell
def __(f, mo):
    if f.name() is not None:
        _output = mo.md("""### **Model selection and training**""")
    else:
        _output = None
    _output
    return


@app.cell
def __(mo, trained, training):
    _output = None
    if training:
        _output = mo.md("Training...")
    if trained:
        _output = mo.md("Training finished")
    _output
    return


@app.cell
def __(f, features, gs, sources_column, train):
    trained = False
    training = False
    if f.name() is not None:
        if sources_column.value is not None:
            training = True
            # Load training data
            _X = train[features]
            _y = train[sources_column.value]
        
            # Perform the grid search
            models = {}
            for _g in gs:
                gs[_g].fit(_X, _y)
                models[_g] = gs[_g].best_estimator_
        
            trained = True
            training = False
    return models, trained, training


@app.cell
def __(mo, trained):
    _output = None
    if trained:
        _output = mo.md("""### **Prediction**""")
    _output
    return


@app.cell
def __(mo, trained):
    test_file = mo.ui.file(label="Uncorrelated tephras")
    if trained:
        _output = mo.md(f"""
            Upload the csv file with the tephras to be correlated.  
            {test_file}
        """)
    else:
        _output = None
    _output
    return test_file,


@app.cell
def __(all_features, io, mo, np, pd, test_file):
    if test_file.name() is not None:
        test_df       = pd.read_csv(io.StringIO(test_file.contents().decode()))
        _non_features = list(set(test_df.select_dtypes(exclude=np.number).columns.tolist()).difference(set(all_features)))
        samples_col   = mo.ui.radio(options=_non_features, inline=True, label="**Groups column**: ")
    return samples_col, test_df


@app.cell
def __(mo, samples_col, test_file):
    if test_file.name() is not None:
        _output = mo.md(f"""
            Select the column of the file containing the sources.
            {samples_col}
        """)
    else:
        _output = None
    _output
    return


@app.cell
def __(mo, predicted, trained):
    if trained and predicted:
        _output = mo.md("""
            Download the table below for inspection of the results.
        """)
    else:
        _output = None
    _output
    return


@app.cell
def __(
    features,
    gs,
    label_encoder,
    models,
    np,
    prediction_averages,
    samples_col,
    test_df,
    trained,
):
    predicted = False
    if (trained) and (samples_col.value is not None):
        _preds = test_df
        _X    = _preds[features]
        label_encoder.fit(models["rf"].classes_)

        # Perform the grid search
        _ys = {}
        for model in models:
            _ys[model] = models[model].predict(_X)

        # Validate the models
        _nn_probs  =  gs["nn"].best_estimator_.predict_proba(_X)
        _rf_probs  =  gs["rf"].best_estimator_.predict_proba(_X)
        _avg_probs = (_rf_probs + _nn_probs) / 2
        _ys["avg"] = label_encoder.inverse_transform(np.argmax(_avg_probs, axis=1))

        # Add the predictions of each classifier to the data frame
        for model, y in _ys.items():
            _preds.loc[:, model] = y

        # Add the predicted probabilities to the the data frame
        _models = [('RF', _rf_probs), ('NN', _nn_probs), ('AVG', _avg_probs)]
        _labels = models["rf"].classes_
        for name, probs in _models:
            columns = [f"{label}_{name}" for label in _labels]
            _preds[columns] = probs

        _preds[_preds[samples_col.value].endswith("average")] = prediction_averages(_preds, samples_col.value, _models)
        predicted = True
    else:
        _preds = None
    _preds
    return columns, model, name, predicted, probs, y


@app.cell
def __(average, end):
    def prediction_averages(df, samples_col, models):
        avgs = df[df[samples_col].endswith("average")]
        labels = models.keys()
        probs = ["RF", "NN", "AVG"]
        
        for i in range(len(avgs)):
            avg = avgs.Comment[i]
            sample = avg[0:avg.rfind('-') - 1]
            temp = df[ (df[samples_col].startswith(sample)) & (~df[samples_col].endswith(average)), : ]
            avgs[i, labels] = temp[labels].mode()
            avgs[i, probs] = temp[probs].mean()
        end
        return avgs
    return prediction_averages,


@app.cell
def __():
    all_features = ["Na2O", "K2O", "FeO", "SiO2", "TiO2", "MgO", "CaO", "MnO", "Al2O3", "P2O5", "SI/K", "Mg/Ca", "Fe/Mg",
        "Fe/Ti", "Fe/Si", "Fe/Ca", "Ti/K20", "SI/TI", "Li7", "Mg24", "Al27", "P31", "Ca43", "Sc45",
        "Ti47", "V51", "Cr52", "Mn55", "Co59", "Cu63", "Zn66", "Ga69", "Rb85", "Sr88", "Y89", "Zr90", "Nb93", "Sn118",
        "Cs133", "Ba137", "La139", "Ce140", "Pr141", "Nd146", "Sm147", "Eu153", "Gd157", "Tb159", "Dy163", "Ho165", "Er166", "Tm169",
        "Yb172", "Lu175", "Hf178", "Ta181", "Pb208", "Th232", "U238", "La/Th", "La/Yb", "Rb/La", "La/Sm", "Ce/Yb",
        "Zr/Cs", "Pb/Nd", "Li/Y", "Nb/Rb", "Ba/Ce", "Ce/Pb", "U/Th", "Ba/La", "U/Pb", "Ba/Rb",
        "Nb/Ta", "Ba/Th", "Th/Ta", "Ba/Nb", "Nb/Th", "Rb/Hf", "Rb/Nd", "Rb/Sr", "Ba/Zr", "Ti/Zr",
        "Zr/Nb", "Dy/Lu", "La/Nb", "Sm/Yb", "Ta/U", "Nb/Zr", "Ce/Eu", "Ce/Hf", "Sm/Nd", "Sm/Zr", "Lu/Hf", "K/La",
        "Rb/Th", "Nb/Ti", "Zr/Hf", "Nb/U", "Rb/Cs", "Ce/U", "Sr/Nd", "U/La"]
    return all_features,


if __name__ == "__main__":
    app.run()
