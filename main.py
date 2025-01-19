
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier # Importing the algorithm
from sklearn.metrics import accuracy_score # importing "accuracy_score" from "sklearn.metrics"


from func import  prep_churn_train, prep_churn_new,  split_churn_train, split_churn_new, training_random_forest, prediction_random_forest, random_forest_feature_importance

def main():
    churn_train_orig = pd.read_csv('churn.csv')
    churn_train_orig.shape
    
    churn_train_orig.columns
    
    churn_train_orig
    
    churn_train_orig.columns
    
    churn_new_orig = pd.read_csv('collection.csv',index_col=0)
    churn_new_orig.shape
    
    churn_new_orig.columns
    
    churn_train = prep_churn_train(churn_train_orig)
    
    print(churn_train.info())
    
    churn_new = prep_churn_new(churn_new_orig)
    
    print(churn_new.info())
    
    x_train, y_train, cstm_train = split_churn_train(churn_train)
    
    x_new, cstm_new = split_churn_new(churn_new)
    
    model = training_random_forest(100,3,1,x_train,y_train)
    
    churn_new_orig_with_predict = prediction_random_forest(model,x_new,churn_new_orig)
    churn_new_orig_with_predict.to_csv('churn_new_orig_with_predict.csv')
    
main()