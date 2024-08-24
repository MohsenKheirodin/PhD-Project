import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import requests
import matplotlib.pyplot as plt
import shap
import copy
from tqdm import tqdm
from hmmlearn import hmm
from sklearn.pipeline import make_pipeline
# comment
# Custom Output Functions

def custom_loss(preds, dtrain, Mat_W):
    labels = dtrain.get_label()
    num_classes = len(np.unique(labels))
    preds = preds.reshape(-1, num_classes)  # Reshape preds to (num_rows, num_classes)
    
    # Apply softmax to get probabilities
    preds_prob = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
    
    labels_one_hot = np.eye(num_classes)[labels.astype(int)]

    # preds_prob = np.array([
    # [0.08576079, 0.18179614, 0.73244307],
    # [0.14431261, 0.03664219, 0.81904519],
    # [0.06709282, 0.49670749, 0.43619969]
    # ])

    # labels_one_hot = np.array([
    # [1.0, 0.0, 0.0],
    # [0.0, 0.0, 1.0],
    # [0.0, 1.0, 0.0]
    # ])
    
    grad = preds_prob - labels_one_hot
    
    if Mat_W is not None:
        coef = copy.deepcopy(labels_one_hot)
        for i in range(len(labels_one_hot)):
            if labels_one_hot[i,0] == 1:
                coef[i,0] = Mat_W[0,0]
                coef[i,1] = Mat_W[0,1]
                coef[i,2] = Mat_W[0,2]
                coef[i,3] = Mat_W[0,3]
            if labels_one_hot[i,1] == 1:
                coef[i,0] = Mat_W[1,0]
                coef[i,1] = Mat_W[1,1]
                coef[i,2] = Mat_W[1,2]
                coef[i,3] = Mat_W[1,3]
            if labels_one_hot[i,2] == 1:
                coef[i,0] = Mat_W[2,0]
                coef[i,1] = Mat_W[2,1]
                coef[i,2] = Mat_W[2,2]
                coef[i,3] = Mat_W[2,3]
            if labels_one_hot[i,3] == 1:
                coef[i,0] = Mat_W[3,0]
                coef[i,1] = Mat_W[3,1]
                coef[i,2] = Mat_W[3,2]
                coef[i,3] = Mat_W[3,3]
            
        grad = grad * coef
    
    hess = preds_prob * (1 - preds_prob)
    
    grad = grad.flatten().astype(np.float32)
    hess = hess.flatten().astype(np.float32)
    
    return grad, hess


# Decision Tree model

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE

def Decision_tree(X_train, X_test, y_train, y_test ,max_depth):
    
    dt = DecisionTreeClassifier(max_depth = max_depth, random_state = 1)
    dt.fit(X_train, y_train)
    
    y_pred = dt.predict(X_test)
    y_pred_train = dt.predict(X_train)
    y_pred_proba_decision_tree = dt.predict_proba(X_test)
    
    print('Accuracy_train = ' + str(accuracy_score(y_pred_train, y_train)))
    print('Accuracy_test = ' + str(accuracy_score(y_pred, y_test)))
    importances = dt.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': dt.feature_importances_
    })
    importance_df.sort_values(by = 'Importance', ascending = False, inplace = True)
    
    return y_pred, y_pred_proba_decision_tree, importance_df, dt 

# Decision Tree model with feature selection using Recursive Feature Elimination (RFE) 
    
def Decision_tree_RFE(X_train, X_test, y_train, y_test, max_depth, num_feat):
    
    dt = DecisionTreeClassifier(max_depth = max_depth, random_state = 1)
    rfe = RFE(dt, n_features_to_select=num_feat)
    rfe = rfe.fit(X_train, y_train)
    
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)
    
    dt.fit(X_train_rfe, y_train)
    y_pred = dt.predict(X_test_rfe)
    y_pred_train = dt.predict(X_train_rfe)
    y_pred_proba_decision_tree = dt.predict_proba(X_test_rfe)

    print(f"Accuracy_train: {accuracy_score(y_pred_train, y_train) * 100:.2f}%")
    print(f"Accuracy_test: {accuracy_score(y_pred, y_test) * 100:.2f}%")
    
    importances = dt.feature_importances_
    
    selected_features = rfe.support_
    feature_ranking = rfe.ranking_
    features = X_train.columns
    importance_decision_tree_df = pd.DataFrame(feature_ranking, index=features, columns=['Ranking'])
    return y_pred, y_pred_proba_decision_tree, importance_decision_tree_df, dt
    
    
# # XGBoost Model

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from xgboost import XGBClassifier, XGBRegressor

def XGB(X_train, X_test, y_train, y_test, is_binary, mode='classification', Weight_out=None, Weight_input=None, Bagging=False, num_boost=50, alpha=0, lambda_=1, show_result = True):
    
    Bagging_params = {
        'colsample_bytree': 0.8,  # 80% of features are randomly chosen for each tree
        'colsample_bylevel': 0.9,  # 90% of features are randomly chosen for each level
        'colsample_bynode': 0.8    # 80% of features are randomly chosen for each split
    }
    
    if mode == 'regression':
        params = {
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_estimators': num_boost,
            'objective': 'reg:squarederror',
            'alpha': alpha,  # L1 regularization term on weights
            'lambda': lambda_  # L2 regularization term on weights
        }

        if Bagging:
            params.update(Bagging_params)
        
        xgb_model = XGBRegressor(**params)
        if Weight_input is not None:
            xgb_model.fit(X_train, y_train, sample_weight=Weight_input, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
        else:
            xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)

        y_pred2 = xgb_model.predict(X_test)
        y_pred2_train = xgb_model.predict(X_train)
        y_proba_xgb =y_pred2
        y_proba_xgb_train = y_pred2_train  # For regression, predictions are probabilities

    elif is_binary:
        params = {
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_estimators': num_boost,
            'objective': 'binary:logistic',
            'alpha': alpha,  # L1 regularization term on weights
            'lambda': lambda_  # L2 regularization term on weights
        }

        if Bagging:
            params.update(Bagging_params)
        
        xgb_model = XGBClassifier(**params)
        if Weight_input is not None:
            xgb_model.fit(X_train, y_train, sample_weight=Weight_input, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
        else:
            xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)

        y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
        y_proba_xgb_train = xgb_model.predict_proba(X_train)[:, 1]
        y_pred2 = np.round(y_proba_xgb)
        y_pred2_train = np.round(y_proba_xgb_train)
        
                
    else:
        num_class = len(np.unique(y_train))
        params = {
            'max_depth': 3,
            'learning_rate': 0.1,
            'objective': 'multi:softprob',
            'num_class': num_class,
            'alpha': alpha,  # L1 regularization term on weights
            'lambda': lambda_,  # L2 regularization term on weights
            'eval_metric': 'mlogloss'
        }

        if Bagging:
            params.update(Bagging_params)

        if Weight_out is not None:
            if Weight_input is not None:
                dtrain = xgb.DMatrix(X_train, label=y_train, weight=Weight_input)
            else:
                dtrain = xgb.DMatrix(X_train, label=y_train)
                
            dtest = xgb.DMatrix(X_test, label=y_test)
            
            def wrapped_custom_loss(preds, dtrain):
                return custom_loss(preds, dtrain, Weight_out)
                
            xgb_model = xgb.train(params, dtrain, num_boost_round=num_boost, obj=wrapped_custom_loss, evals=[(dtest, 'test')], early_stopping_rounds=10, verbose_eval=False)
            y_pred2 = np.argmax(xgb_model.predict(dtest), axis=1)
            y_pred2_train = np.argmax(xgb_model.predict(dtrain), axis=1)
            y_proba_xgb = xgb_model.predict(dtest)
            
        else:
            xgb_model = XGBClassifier(**params)
            if Weight_input is not None:
                xgb_model.fit(X_train, y_train, sample_weight=Weight_input, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
            else:
                xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
        
            y_pred2 = np.argmax(xgb_model.predict_proba(X_test), axis=1)
            y_pred2_train = np.argmax(xgb_model.predict_proba(X_train), axis=1)
            y_proba_xgb = xgb_model.predict_proba(X_test)
            y_proba_xgb_train = xgb_model.predict_proba(X_train)
    
    # Feature importance
    if Weight_out is not None and not is_binary and mode != 'regression':
        importance = xgb_model.get_score(importance_type='weight')
    else:
        importance = xgb_model.get_booster().get_score(importance_type='weight')
    
    importance_XGB_df = pd.DataFrame({
        'Feature': importance.keys(),
        'Importance': importance.values()
    })
    importance_XGB_df.sort_values(by='Importance', ascending=False, inplace=True)

    if mode == 'regression':
        from sklearn.metrics import mean_squared_error
        accuracy_train = mean_squared_error(y_train, y_pred2_train, squared=False)
        accuracy_test = mean_squared_error(y_test, y_pred2, squared=False)
        if show_result:
            print(f"RMSE_train: {accuracy_train:.2f}")
            print(f"RMSE_test: {accuracy_test:.2f}")
    else:
        accuracy_train = accuracy_score(y_train, y_pred2_train)
        accuracy_test = accuracy_score(y_test, y_pred2)
        if show_result:
            print(f"Accuracy_train: {accuracy_train * 100:.2f}%")
            print(f"Accuracy_test: {accuracy_test * 100:.2f}%")

    return y_pred2, y_proba_xgb, y_pred2_train, y_proba_xgb_train, importance_XGB_df, xgb_model, [accuracy_train, accuracy_test]


def XGB_RFE(X_train, X_test, y_train, y_test, mode, num_feat):

    if mode == 'binary':
        xgb_model = xgb.XGBClassifier(objective='binary:logistic')
    else:
        xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4)

    rfe = RFE(xgb_model, n_features_to_select=num_feat)
    rfe = rfe.fit(X_train, y_train)

    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)
    
    xgb_model.fit(X_train_rfe, y_train)
    
    y_pred2 = xgb_model.predict(X_test_rfe)
    y_pred2_train = xgb_model.predict(X_train_rfe)
    y_proba_xgb = xgb_model.predict_proba(X_test_rfe)
    
    importances = xgb_model.feature_importances_
    
    accuracy_train = accuracy_score(y_train,  y_pred2_train)
    accuracy_test = accuracy_score(y_test, y_pred2)
    
    print(f"Accuracy_train: {accuracy_train * 100:.2f}%")
    print(f"Accuracy_test: {accuracy_test * 100:.2f}%")

    selected_features = rfe.support_
    feature_ranking = rfe.ranking_
    features = X_train.columns
    importance_df = pd.DataFrame(feature_ranking, index=features, columns=['Ranking'])

    
    return y_pred2, y_proba_xgb, importance_df, xgb_model


# XGB Fine Tuning

def xgb_fine_tune_stock_level(X_train, X_test, y_train, y_test, model_name, n_estimators=30, learning_rate=0.01):
    
    y_pred_xgb_multi_tuned = pd.DataFrame()
    y_proba_xgb_multi_tuned = pd.DataFrame()
    list_stocks = X_train['symbol'].unique().tolist()
    
    for stock_name in list_stocks[0:]:
        print(stock_name)
        
        X_train_stock = X_train[X_train['symbol'] == stock_name]
        y_train_stock = y_train[y_train['symbol'] == stock_name]
        X_test_stock  = X_test[X_test['symbol'] == stock_name]
    
        fine_tuned_model = xgb.XGBClassifier()
        fine_tuned_model.load_model(model_name)
        fine_tuned_model.set_params(n_estimators=n_estimators, learning_rate=learning_rate)
        fine_tuned_model.fit(X_train_stock.iloc[:,1:], y_train_stock['label_four'] - 1)
    
        y_pred_xgb_multi_seperate = pd.DataFrame(fine_tuned_model.predict(X_test_stock.iloc[:,1:]), index=X_test_stock.index, columns=['label_predicted'])
        y_proba_xgb_multi_seperate = pd.DataFrame(fine_tuned_model.predict_proba(X_test_stock.iloc[:,1:]), index=X_test_stock.index)
        y_pred_xgb_multi_seperate['symbol'] = stock_name
        y_proba_xgb_multi_seperate['symbol'] = stock_name
    
        y_pred_xgb_multi_tuned = pd.concat([y_pred_xgb_multi_tuned, y_pred_xgb_multi_seperate], axis=0)
        y_proba_xgb_multi_tuned = pd.concat([y_proba_xgb_multi_tuned, y_proba_xgb_multi_seperate], axis=0)
    
    list_stocks_test = X_test['symbol'].unique().tolist()
    list_stocks_after_split = [item for item in list_stocks_test if item not in list_stocks]   
    
    for stock_name in list_stocks_after_split[0:]:
        print(stock_name)
        X_test_stock  = X_test[X_test['symbol'] == stock_name]
        
        y_pred_xgb_multi_seperate = pd.DataFrame(xgb_model_multi.predict(X_test_stock.iloc[:,1:]), index=X_test_stock.index, columns=['label_predicted'])
        y_proba_xgb_multi_seperate = pd.DataFrame(xgb_model_multi.predict_proba(X_test_stock.iloc[:,1:]), index=X_test_stock.index)
        y_pred_xgb_multi_seperate['symbol'] = stock_name
        y_proba_xgb_multi_seperate['symbol'] = stock_name
    
        y_pred_xgb_multi_tuned = pd.concat([y_pred_xgb_multi_tuned, y_pred_xgb_multi_seperate], axis=0)
        y_proba_xgb_multi_tuned = pd.concat([y_proba_xgb_multi_tuned, y_proba_xgb_multi_seperate], axis=0)

    return y_pred_xgb_multi_tuned, y_proba_xgb_multi_tuned


import xgboost as xgb

def xgb_fine_tune(X_train, X_test, y_train, y_test, model_name, mode='regression', n_estimators=30, learning_rate=0.01):
    if mode == 'classification':
        fine_tuned_model = xgb.XGBClassifier()
    elif mode == 'regression':
        fine_tuned_model = xgb.XGBRegressor()
    else:
        raise ValueError("Invalid mode. Choose 'classification' or 'regression'.")

    fine_tuned_model.load_model(model_name)
    fine_tuned_model.set_params(n_estimators=n_estimators, learning_rate=learning_rate)
    fine_tuned_model.fit(X_train, y_train)
    y_pred_xgb_tuned = fine_tuned_model.predict(X_test)

    if mode == 'classification':
        y_proba_xgb_tuned = fine_tuned_model.predict_proba(X_test)
    else:
        y_proba_xgb_tuned = y_pred_xgb_tuned

    return y_pred_xgb_tuned, y_proba_xgb_tuned
    
    

def XGBooster(X_train, X_test, y_train, y_test, num_round=10, num_class=4):

    y_train = y_train - y_train.min()
    y_test = y_test - y_test.min()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'max_depth': 3,
        'eta': 0.1,
        'objective': 'multi:softprob',  # Use multi:softprob to get class probabilities
        'eval_metric': 'mlogloss',
        'num_class': num_class  # Specify the number of classes
    }

    bst = xgb.train(params, dtrain, num_round)
    
    y_proba_xgb = bst.predict(dtest)
    y_proba_train = bst.predict(dtrain)
    
    y_pred2 = np.argmax(y_proba_xgb, axis=1)
    y_pred2_train = np.argmax(y_proba_train, axis=1)

    importance = bst.get_score(importance_type='weight')
    importance_xgb = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    })
    importance_xgb.sort_values(by='Importance', ascending=False, inplace=True)

    test_accuracy = accuracy_score(y_test, y_pred2)
    train_accuracy = accuracy_score(y_train, y_pred2_train)
    print(f"Train set accuracy: {train_accuracy:.2f}")
    print(f"Test set accuracy: {test_accuracy:.2f}")

    return y_pred2, y_proba_xgb, y_proba_train, importance_xgb, bst


def XGBooster_finetune(X_train, X_test, y_train, y_test, model, num_round=10, num_class=4):
    y_train = y_train - y_train.min()
    y_test = y_test - y_test.min()

    dtrain_new = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    bst = xgb.Booster()
    bst.load_model(model)
    bst.update(dtrain_new, num_round)

    y_proba_xgb = bst.predict(dtest)
    y_proba_train = bst.predict(dtrain_new)
    
    y_pred2 = np.argmax(y_proba_xgb, axis=1)
    y_pred2_train = np.argmax(y_proba_train, axis=1)

    test_accuracy = accuracy_score(y_test, y_pred2)
    train_accuracy = accuracy_score(y_train, y_pred2_train)
    print(f"Train set accuracy: {train_accuracy:.2f}")
    print(f"Test set accuracy: {test_accuracy:.2f}")

    return y_pred2, y_proba_xgb


# Bagging Classifier

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

def Bagging(X_train, X_test, y_train, y_test, is_binary = False, mode='classification', n_estimators=50):
    base_estimator = DecisionTreeClassifier()
    bagging_clf = BaggingClassifier(base_estimator=base_estimator, n_estimators=n_estimators, random_state=42)
    
    bagging_clf.fit(X_train, y_train)
    y_pred = bagging_clf.predict(X_test)
    y_proba = bagging_clf.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    feature_importances = np.mean([tree.feature_importances_ for tree in bagging_clf.estimators_], axis=0)
    importance_df = pd.DataFrame({
        'Feature': [f'feature_{i}' for i in range(X_train.shape[1])],
        'Importance': feature_importances
    })
    importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    
    return y_pred, y_proba, importance_df, bagging_clf


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import logging


def LSTM_model(X_train_ts, X_test_ts, y_train_ts, y_test_ts, lags, is_binary, label, epoch = 10):
    
    combined_X_train, combined_X_test, combined_y_train, combined_y_test, y_test_ts, X_test_ts, label_encoder = Make_Data_Ready_for_LSTM(X_train_ts, X_test_ts, y_train_ts, y_test_ts, 
                                                                                                                                         label = label, lags = lags, norm = 'minmax')
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(lags, combined_X_train.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(4, activation='sigmoid'))

    if is_binary:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        

    model.fit(combined_X_train, combined_y_train, epochs=epoch, batch_size=32, validation_data=(combined_X_test, combined_y_test))
    loss, accuracy = model.evaluate(combined_X_test, combined_y_test)
    print(f'Loss: {loss}, Accuracy: {accuracy}')
    y_pred_proba = model.predict(combined_X_test)
    
    
    if is_binary:
        y_pred = (y_pred_proba > 0.5).astype("int32")
        y_test_labels = y_test        
    else:
        y_pred = label_encoder.inverse_transform(np.argmax(y_pred_proba, axis=1))
        y_test_labels = label_encoder.inverse_transform(np.argmax(combined_y_test, axis=1))
                                                        

    accuracy = accuracy_score(y_test_labels, y_pred)
    conf_matrix = confusion_matrix(y_test_labels, y_pred)
    class_report = classification_report(y_test_labels, y_pred)
    print(f'Accuracy: {accuracy}')
    
    return y_pred, y_pred_proba, model, X_test_ts, y_test_ts



# LSTM Fine Tuning

def lstm_fine_tune_stock_level(X_train, X_test, y_train, y_test, epochs=10, batch_size=32):

    y_pred_lstm_multi_tuned = pd.DataFrame()
    y_proba_lstm_multi_tuned = pd.DataFrame()
    list_stocks = X_train['symbol'].unique().tolist()
    
    for stock_name in list_stocks[0:10]:
        print(stock_name)
        
        X_train_stock = X_train[X_train['symbol'] == stock_name]
        y_train_stock = y_train[y_train['symbol'] == stock_name]
        X_test_stock  = X_test[X_test['symbol'] == stock_name]
        test_index = X_test_stock.index
    
        X_train_stock = X_train_stock.iloc[:,1:].values
        X_test_stock  = X_test_stock.iloc[:,1:].values
        y_train_stock = y_train_stock['label_four'] - 1
        
        onehot_encoder = OneHotEncoder(sparse_output=False)
        y_train_stock_encoded = onehot_encoder.fit_transform(y_train_stock.values.reshape(-1, 1))
    
        X_train_stock_reshaped = X_train_stock.reshape((X_train_stock.shape[0], 1, X_train_stock.shape[1]))
        X_test_stock_reshaped = X_test_stock.reshape((X_test_stock.shape[0], 1, X_test_stock.shape[1]))
    
        model_lstm_multi_stock = load_model('global_lstm_multi_model.h5')
        # for layer in model_lstm_multi_stock.layers[:-1]:  # Freeze all layers except the last one
        #     layer.trainable = False
        model_lstm_multi_stock.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model_lstm_multi_stock.fit(X_train_stock_reshaped, y_train_stock_encoded, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
        y_pred_proba = model_lstm_multi_stock.predict(X_test_stock_reshaped)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        y_pred_lstm_multi_seperate = pd.DataFrame(y_pred,        index=test_index, columns=['label_predicted'])
        y_proba_lstm_multi_seperate = pd.DataFrame(y_pred_proba, index=test_index)
       
        y_pred_lstm_multi_seperate['symbol'] = stock_name
        y_proba_lstm_multi_seperate['symbol'] = stock_name
    
        y_pred_lstm_multi_tuned = pd.concat([y_pred_lstm_multi_tuned, y_pred_lstm_multi_seperate], axis=0)
        y_proba_lstm_multi_tuned = pd.concat([y_proba_lstm_multi_tuned, y_proba_lstm_multi_seperate], axis=0)
    
    
    list_stocks_test = X_test['symbol'].unique().tolist()
    list_stocks_after_split = [item for item in list_stocks_test if item not in list_stocks]   
    
    for stock_name in list_stocks_after_split[0:]:
        print(stock_name)
        model_lstm_multi_stock = load_model('global_lstm_multi_model.h5')
        X_test_stock  = X_test[X_test['symbol'] == stock_name]
        test_index = X_test_stock.index
        X_test_stock = X_test_stock.iloc[:,1:].values
    
        X_test_stock_reshaped = X_test_stock.reshape((X_test_stock.shape[0], 1, X_test_stock.shape[1]))
        
        y_pred_proba = model_lstm_multi_stock.predict(X_test_stock_reshaped)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        y_pred_lstm_multi_seperate = pd.DataFrame(y_pred,        index=test_index, columns=['label_predicted'])
        y_proba_lstm_multi_seperate = pd.DataFrame(y_pred_proba, index=test_index)
        
        y_pred_lstm_multi_seperate['symbol'] = stock_name
        y_proba_lstm_multi_seperate['symbol'] = stock_name
    
        y_pred_lstm_multi_tuned = pd.concat([y_pred_lstm_multi_tuned, y_pred_lstm_multi_seperate], axis=0)
        y_proba_lstm_multi_tuned = pd.concat([y_proba_lstm_multi_tuned, y_proba_lstm_multi_seperate], axis=0)

    return y_pred_lstm_multi_tuned, y_proba_lstm_multi_tuned
    

def lstm_fine_tune(X_train_ts, X_test_ts, y_train_ts, y_test_ts, lags, model, label, epochs=10, batch_size=32):

    combined_X_train, combined_X_test, combined_y_train, combined_y_test, y_test_ts, X_test_ts, label_encoder = Make_Data_Ready_for_LSTM(X_train_ts, X_test_ts, y_train_ts, y_test_ts, 
                                                                                                                                         label = label, lags = lags, norm = 'minmax')
    
    model_lstm_multi_fined = load_model(model)
     # for layer in model_lstm_multi_fined.layers[:-1]:  # Freeze all layers except the last one
        #     layer.trainable = False
    model_lstm_multi_fined.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model_lstm_multi_fined.fit(combined_X_train, combined_y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    y_pred_proba = model_lstm_multi_fined.predict(combined_X_test)        
    y_pred = label_encoder.inverse_transform(np.argmax(y_pred_proba, axis=1))
    y_test_labels = label_encoder.inverse_transform(np.argmax(combined_y_test, axis=1))

    return y_pred, y_pred_proba, X_test_ts, y_test_ts



# Random Forest

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def Random_Forest(X_train, X_test, y_train, y_test):
    
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    y_pred_proba = rf_clf.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f'Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)

    feature_importances = rf_clf.feature_importances_
    features = X_train.columns.tolist()
    importance_df = pd.DataFrame({'Feature': features,'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    return y_pred, y_pred_proba, importance_df, rf_clf


# Stochastic Gradient Decent Classifier

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report

def SGD_Classifier(X_train, X_test, y_train, y_test, max_iter=1000, tol=1e-3, alpha=0.0001, loss='log_loss', learning_rate='optimal'):

    model = make_pipeline(
        StandardScaler(),
        SGDClassifier(max_iter=max_iter, tol=tol, alpha=alpha, loss=loss, learning_rate=learning_rate, random_state=42)
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    log_loss_value = log_loss(y_test, y_proba)
    print(f"Log Loss: {log_loss_value:.2f}")
    
    print("Classification Report:")
    # print(classification_report(y_test, y_pred))

    feature_importances = model.named_steps['sgdclassifier'].coef_[0]  # For binary classification, get coefficients of the first class
    importance_df = pd.DataFrame({
        'Feature': [f'feature_{i}' for i in range(X_train.shape[1])],
        'Importance': feature_importances})
    importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    
    return y_pred, y_proba, importance_df, model


# Support Vector Machine Model
 
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def Support_Vector_Machine(X_train, X_test, y_train, y_test):
    
    svm_clf = SVC(probability=True, kernel='linear', random_state=42)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)
    y_pred_proba = svm_clf.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f'Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)
    
    print('Prediction Probabilities for the first 5 samples:')
    print(y_pred_proba[:5])
    
    # Handling feature names
    if isinstance(X_train, pd.DataFrame):
        features = X_train.columns.tolist()
    else:
        features = [f'Feature {i}' for i in range(X_train.shape[1])]
    
    # For binary classification, svm_clf.coef_ has a shape (1, n_features)
    if len(svm_clf.coef_) == 1:
        feature_importances = np.abs(svm_clf.coef_[0])
    else:
        # For multi-class, we average the absolute values of the coefficients across classes
        feature_importances = np.mean(np.abs(svm_clf.coef_), axis=0)
    
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    print('Feature Importances:')
    print(importance_df)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importances (Approximated)')
    plt.show()

    return y_pred, y_pred_proba, importance_df, svm_clf



# Artificial Neural Network (ANN)

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.inspection import permutation_importance

def Artificial_Neural_Network(X_train, X_test, y_train, y_test, is_binary=False):
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = Sequential()
    model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    
    if is_binary:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.add(Dense(len(np.unique(y_train)), activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
    y_pred_proba = model.predict(X_test)
    
    if is_binary:
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f'Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)
    
    print('Prediction Probabilities for the first 5 samples:')
    print(y_pred_proba[:5])
    
    # Get feature importances using permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, scoring='accuracy')
    
    features = X_train.columns if isinstance(X_train, pd.DataFrame) else [f'Feature {i}' for i in range(X_train.shape[1])]
    importance_df = pd.DataFrame({'Feature': features, 'Importance': perm_importance.importances_mean})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # print('Feature Importances:')
    # print(importance_df)
    
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x='Importance', y='Feature', data=importance_df)
    # plt.title('Feature Importances (Permutation Importance)')
    # plt.show()

    return y_pred, y_pred_proba, importance_df, model



# Transformer Model

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, Embedding
from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

def Transformer_model(X_train_ts, X_test_ts, y_train_ts, y_test_ts, lags, is_binary, label, epoch=10):
    
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        x = Dense(ff_dim, activation="relu")(res)
        x = Dropout(dropout)(x)
        x = Dense(inputs.shape[-1])(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        return x + res

    def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
        inputs = Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(mlp_dropout)(x)
        outputs = Dense(4, activation="sigmoid" if is_binary else "softmax")(x)
        return Model(inputs, outputs)
    
    # Prepare data as done in LSTM_model
    combined_X_train, combined_X_test, combined_y_train, combined_y_test, y_test_ts, X_test_ts, label_encoder = Make_Data_Ready_for_LSTM(X_train_ts, X_test_ts, y_train_ts, y_test_ts, label=label, lags=lags, norm='minmax')
    
    input_shape = combined_X_train.shape[1:]
    model = build_model(
        input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25
    )
    
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy' if is_binary else 'categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(combined_X_train, combined_y_train, epochs=epoch, batch_size=32, validation_data=(combined_X_test, combined_y_test))
    loss, accuracy = model.evaluate(combined_X_test, combined_y_test)
    print(f'Loss: {loss}, Accuracy: {accuracy}')
    
    y_pred_proba = model.predict(combined_X_test)
    
    if is_binary:
        y_pred = (y_pred_proba > 0.5).astype("int32")
        y_test_labels = y_test_ts        
    else:
        y_pred = label_encoder.inverse_transform(np.argmax(y_pred_proba, axis=1))
        y_test_labels = label_encoder.inverse_transform(np.argmax(combined_y_test, axis=1))

    accuracy = accuracy_score(y_test_labels, y_pred)
    conf_matrix = confusion_matrix(y_test_labels, y_pred)
    class_report = classification_report(y_test_labels, y_pred)
    print(f'Accuracy: {accuracy}')
    
    return y_pred, y_pred_proba, model, X_test_ts, y_test_ts