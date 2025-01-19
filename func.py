
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier # Importing the algorithm
from sklearn.metrics import accuracy_score # importing "accuracy_score" from "sklearn.metrics"

def prep_churn_train (df):

    df = df.rename(columns=str.lower) # Rename columns to lower letters
    df.churn = (df.churn=='Yes').astype('int') # Label to numeric
    df['totalcharges'] = df['totalcharges'].replace({" ": "0.0"}).astype(float)
    df=df.rename_axis('custid').reset_index()
    # df['custid']=df['customerid'].str.strip().str[4:].astype(float)
    df = df.drop(['paperlessbilling',  'customerid'], axis=1) # Drop some features which aren't informative
    df = df.drop(['phoneservice','multiplelines','internetservice','onlinesecurity','onlinebackup',
               'deviceprotection','techsupport','streamingtv','streamingmovies'], axis=1)
    df = pd.get_dummies(df) # Categorical values to 1-hot ("one hot" encoding is a representation of categorical variables as binary vectors)
    
    df = df.astype(float) # Let's convert all data to float because some modules warn against other types
    
    
    print('')
    
    # Check for nulls
    null = df.isna().sum().sum()
    print (f'There are {null} null values')
    
    # Check for DataType
    dtype = df.dtypes.unique()[0]
    print (f'There are {dtype} dtype')
    
    print('')
    

    return df



def prep_churn_new (df):

    df = df.rename(columns=str.lower) # Rename columns to lower letters

    df['totalcharges'] = df['totalcharges'].replace({" ": "0.0"}).astype(float)
    df=df.rename_axis('index').reset_index()
    df['custid']=df['customerid'].str.strip().str[-4:].astype(float)

    df = df.drop(['paperlessbilling', 'customerid','index'], axis=1) # Drop some features which aren't informative
    df = df.drop([ 'services.phoneservice','services.multiplelines','services.internetservice','services.onlinesecurity','services.onlinebackup',
               'services.deviceprotection','services.techsupport','services.streamingtv','services.streamingmovies'], axis=1)
    df = pd.get_dummies(df) # Categorical values to 1-hot ("one hot" encoding is a representation of categorical variables as binary vectors)
   
    df = df.astype(float) # Let's convert all data to float because some modules warn against other types
    
   
    
    
    
    print('')
    
    # Check for nulls
    null = df.isna().sum().sum()
    print (f'There are {null} null values')
    
    # Check for DataType
    dtype = df.dtypes.unique()[0]
    print (f'There are {dtype} dtype')
    
    
    print('')

    return df


def split_churn_train(df):

    label = 'churn'
    cstm = 'custid'

    x_train = df.drop(label, axis=1)
    x_train = x_train.drop(cstm, axis=1)
    y_train = df[label]
    cstm_train = df[cstm]
    
    return x_train,y_train,cstm_train




def split_churn_new(df):
    cstm = 'custid'

    x_new = df.drop(cstm, axis=1)
    cstm_new = df[cstm]
    
    return x_new, cstm_new





def training_random_forest(n,m,r,x_train,y_train):

    model = RandomForestClassifier(n_estimators=n, max_depth=m, random_state=r)
    model.fit(x_train, y_train)
    
    return model




def prediction_random_forest(model,x_new,df_orig):

    y_new = model.predict(x_new) 
    y_new = pd.Series(y_new,name='predict')
    output = df_orig.join(y_new)
    
    return output





def random_forest_feature_importance(model,x_new):

    feature_importances = model.feature_importances_ # applying the method "feature_importances_" on the algorithm
    features = x_new.columns # all the features
    stats = pd.DataFrame({'feature':features, 'importance':feature_importances}) # creating the data frame
    print(stats.sort_values('importance', ascending=False)) # Sorting the data frame

    stats_sort = stats.sort_values('importance', ascending=True)
    stats_sort.plot(y='importance', x='feature', kind='barh')
    plt.title('Feature Importance of Random Forest')
    plt.show()



