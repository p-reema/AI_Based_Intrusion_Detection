# from xgboost import XGBClassifier

# def get_model():
#     return XGBClassifier(
#         n_estimators=500,
#         learning_rate=0.1,
#         max_depth=6,
#         eval_metric='logloss',
#         use_label_encoder=False
#     )


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def get_model(use_grid_search=False):
    base_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss'
    )

    if use_grid_search:
        param_grid = {
            'max_depth': [4, 6],
            'learning_rate': [0.1, 0.2],
            'n_estimators': [100, 200]
        }

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=3,
            verbose=1,
            n_jobs=-1
        )
        return grid_search

    return base_model
