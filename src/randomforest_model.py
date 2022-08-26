from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score
import pandas as pd
import pickle
import preprocessing as pre
import datetime

def get_randomized_search_hyperparameters(X_train:pd.DataFrame,
                                            y_train:pd.DataFrame,
                                            random_grid:dict = {'bootstrap': [True, False],
                                                                'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                                                                'max_features': ['auto', 'sqrt'],
                                                                'min_samples_leaf': [1, 2, 4],
                                                                'min_samples_split': [2, 5, 10],
                                                                'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]},
                                            train_hyperparams:bool = False):

    """
    This function will give us hyperparameters for the Random Forest model.

    Args:
        X_train: pd.DataFrame - The training dataframe.
        y_train: pd.DataFrame - The training labels.
        random_grid:dict - The hyperparameters to be searched.
        train_hyperparams:bool - Whether or not to train the hyperparameters.

    Returns:
        best_hyperparameters: dict - The hyperparameters for the Random Forest model.
    """
    if train_hyperparams == True:
        # random forest model 
        random_forest = RandomForestClassifier()

        # Use the random grid to search for best hyperparameters
        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available

        rf_random = RandomizedSearchCV(estimator = random_forest, param_distributions = random_grid, 
                                        n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

        rf_random.fit(X_train, y_train)

        best_hyperparameters = rf_random.best_params_
    else:
        best_hyperparameters = {'n_estimators': 400,
                                'min_samples_split': 2,
                                'min_samples_leaf': 1,
                                'max_features': 'sqrt',
                                'max_depth': 90,
                                'bootstrap': False}


    return best_hyperparameters



def get_stats(X_train:pd.DataFrame,
              y_train:pd.DataFrame,
              X_test:pd.DataFrame,
              y_test:pd.DataFrame,
              best_hyperparameters:dict,
              classifier:RandomForestClassifier = RandomForestClassifier(),
              pickle_model_name:str = 'Random_Forest',
              pickle_the_model:bool = False):
    """
    This function will train and give us the stats for the Random Forest model.

    Args:
        X_train: pd.DataFrame - The training dataframe.
        y_train: pd.DataFrame - The training labels.
        X_test: pd.DataFrame - The testing dataframe.
        y_test: pd.DataFrame - The testing labels.
        best_hyperparameters: dict - The hyperparameters for the Random Forest model.
        classifier:RandomForestClassifier - The Random Forest model.
        pickle_model_name:str - The name of the model to be pickled.
        pickle_the_model:bool - Whether or not to pickle the model.

    Returns:
        df_stats: pd.DataFrame - The stats for the Random Forest model.
    """
    classifier.set_params(**best_hyperparameters)

    classifier.fit(X_train, y_train)
    
    y_test_pred = classifier.predict(X_test)
    y_test_pred_proba = classifier.predict_proba(X_test)
    
    scores = [(accuracy_score(y_test, y_test_pred),      
               recall_score(y_test, y_test_pred, average='macro'),   # there are differences in scring, keep this in mind
               precision_score(y_test, y_test_pred, average='macro'),
               roc_auc_score(y_test, y_test_pred_proba, average='macro', multi_class='ovo'),
               f1_score(y_test, y_test_pred, average='macro'))]

    column_names = ['Accuracy', 'Recall', 'Precision', 'ROC_AUC' ,'F1_Score']

    df = pd.DataFrame(scores, columns=column_names)
    df_stats = df.round(decimals = 2)

    if pickle_the_model==True:
        now = datetime.datetime.now()
        pickle_model = open('{}_{}_{}.pkl'.format(pickle_model_name, now.month, now.day), 'wb')
        pickle.dump(classifier, pickle_model)
        pickle_model.close()

    return df_stats

def train_model(data:pd.DataFrame,
                pickle_the_model:bool):
    """
    This function will train the Random Forest model.

    Args:
        data:pd.DataFrame - The dataframe to be used for training.
        pickle_the_model:bool - Whether or not to pickle the model.

    Returns:
        df_stats: pd.DataFrame - The stats for the model. Contains the Accuracy, Recall, Precision, ROC_AUC, and F1_Score.
    """
    X_train, X_test, y_train, y_test = pre.preprocess_data_for_training_and_testing(data)
    best_hyperparameters = get_randomized_search_hyperparameters(X_train, y_train)
    df_stats = get_stats(X_train, y_train, X_test, y_test, best_hyperparameters, pickle_the_model=pickle_the_model)

    return df_stats

if __name__ == '__main__':
    data = pd.read_excel("D:\Programing projects\Master_rad\data\Podaci_za_sve_masine.xlsx")
    df_stats = train_model(data, pickle_the_model=True)
