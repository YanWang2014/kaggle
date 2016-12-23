import numpy as np
import pandas as pd
from mydeap import *

def Outputs(data):
    return np.round(1.-(1./(1.+np.exp(-data))))
    
def MungeData(data):
    # Sex
    data.drop(['Ticket', 'Name'], inplace=True, axis=1)
    data.Sex.fillna('0', inplace=True)
    data.loc[data.Sex != 'male', 'Sex'] = 0
    data.loc[data.Sex == 'male', 'Sex'] = 1
    # Cabin
    data.Cabin.fillna('0', inplace=True)
    data.loc[data.Cabin.str[0] == 'A', 'Cabin'] = 1
    data.loc[data.Cabin.str[0] == 'B', 'Cabin'] = 2
    data.loc[data.Cabin.str[0] == 'C', 'Cabin'] = 3
    data.loc[data.Cabin.str[0] == 'D', 'Cabin'] = 4
    data.loc[data.Cabin.str[0] == 'E', 'Cabin'] = 5
    data.loc[data.Cabin.str[0] == 'F', 'Cabin'] = 6
    data.loc[data.Cabin.str[0] == 'G', 'Cabin'] = 7
    data.loc[data.Cabin.str[0] == 'T', 'Cabin'] = 8
    # Embarked
    data.loc[data.Embarked == 'C', 'Embarked'] = 1
    data.loc[data.Embarked == 'Q', 'Embarked'] = 2
    data.loc[data.Embarked == 'S', 'Embarked'] = 3
    data.Embarked.fillna(0, inplace=True)
    data.fillna(-1, inplace=True)
    return data.astype(float)

    

if __name__ == "__main__":
    train = pd.read_csv('Titanic\\train.csv')
    test = pd.read_csv('Titanic\\test.csv')
    mungedtrain = MungeData(train)
    
    #GP
    GeneticFunction = mydeap(mungedtrain)
    
    #test
    mytrain = mungedtrain.iloc[:,2:10].values.tolist()
    trainPredictions = Outputs(np.array([GeneticFunction(*x) for x in mytrain]))

    pdtrain = pd.DataFrame({'PassengerId': mungedtrain.PassengerId.astype(int),
                            'Predicted': trainPredictions.astype(int),
                            'Survived': mungedtrain.Survived.astype(int)})
    pdtrain.to_csv('MYgptrain.csv', index=False)
    from sklearn.metrics import accuracy_score
    print(accuracy_score(mungedtrain.Survived.astype(int),trainPredictions.astype(int)))
    
    mungedtest = MungeData(test)
    mytest = mungedtest.iloc[:,1:9].values.tolist()
    testPredictions = Outputs(np.array([GeneticFunction(*x) for x in mytest]))

    pdtest = pd.DataFrame({'PassengerId': mungedtest.PassengerId.astype(int),
                            'Survived': testPredictions.astype(int)})
    pdtest.to_csv('MYgp.csv', index=False)

