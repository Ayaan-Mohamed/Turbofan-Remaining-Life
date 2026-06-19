def train_and_eval(model, X, y, groups):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_score, recall_score, fbeta_score

    gkf = GroupKFold(n_splits=5)
    scale = StandardScaler()
    precision_total = []
    recall_total = []
    f2_total = []

    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        X_train_scaled = scale.fit_transform(X_train)   # fit + transform
        X_val_scaled   = scale.transform(X_val) 
        model.fit(X_train_scaled, y_train)
        y_predict = model.predict(X_val_scaled)
        precision = precision_score(y_val, y_predict)
        recall = recall_score(y_val, y_predict)
        f2 = fbeta_score(y_val, y_predict, beta=2)
        precision_total.append(precision)
        recall_total.append(recall)
    
        f2_total.append(f2)
    
    print("Precision:", sum(precision_total) / len(precision_total))
    print("Recall:   ", sum(recall_total) / len(recall_total))
    print("f2:       ", np.mean(f2_total))
    print("f2_std:   ",  np.std(f2_total))


    
    return

def optimize_f2(model, X, y, groups, param_grid):

    from sklearn.model_selection import GroupKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import fbeta_score, make_scorer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import numpy as np

    gkf = GroupKFold(n_splits=5)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model),
    ])

    f2_scorer = make_scorer(fbeta_score, beta=2)

    # X = sensor features (unit_number NOT included)
    # y = binary labels (1 = failing, 0 = healthy)
    # groups = the unit_number column

    grid = GridSearchCV(pipe, param_grid, cv = gkf, scoring = f2_scorer)

    grid.fit(X, y, groups=groups)

    print("Winning Parameters", grid.best_params_)
    print("Winning f2 score  ", grid.best_score_)
    print("Spread (std):     ", grid.cv_results_['std_test_score'][grid.best_index_])

    return



