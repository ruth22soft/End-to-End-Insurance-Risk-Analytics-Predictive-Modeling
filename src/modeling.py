from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_models(X_train, y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100),
        'XGBoost': XGBRegressor()
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        results[name] = {
            'RMSE': mean_squared_error(y_test, preds, squared=False),
            'R2': r2_score(y_test, preds)
        }
    return results

def shap_analysis(model, X, feature_names):
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    return shap.summary_plot(
        shap_values, 
        X, 
        feature_names=feature_names,
        plot_type='bar'
    )