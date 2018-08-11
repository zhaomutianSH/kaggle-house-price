import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import time
from sklearn.externals import joblib

TYPE='TEST'
train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')
#读取数据
print(train_df.shape)
print(test_df.shape)
orgain_train=train_df.copy()
orgain_test=test_df.copy()
train_price=np.log1p(train_df.pop('SalePrice'))
train_id=train_df.pop('Id')
test_id=test_df.pop('Id')
#数据整理，抽取多余列
#train_df.shape
print(test_df.shape)
print(train_df.shape)
all_data=pd.concat((train_df,test_df),axis=0)
#合并
all_data['MSSubClass']=all_data['MSSubClass'].astype(str)

#pd.get_dummies(data=all_data['MSSubClass'],prefix='MSSubClass')
#one hot 编码
all_dummies=pd.get_dummies(all_data)
print(all_dummies.shape)

numcols=all_data.columns[all_data.dtypes!='object']
mean_clos=all_dummies.mean()
#缺省值补充
all_dummies_no_nan=all_dummies.fillna(mean_clos)
means=all_dummies_no_nan.loc[:,numcols].mean()
std=all_dummies_no_nan.loc[:,numcols].std()
cleaned_data=all_dummies_no_nan.copy()
cleaned_data.loc[:,numcols]=(cleaned_data.loc[:,numcols]-means)/std
train_cleaned_data=cleaned_data.iloc[0:1460]
test_cleaned_data=cleaned_data.iloc[1460:]
print(train_cleaned_data.shape)
print(test_cleaned_data.shape)
X_trian=train_cleaned_data.values
X_test=test_cleaned_data.values
print(X_trian.shape)
print(train_price.shape)
if TYPE=='TRAIN':
    svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                       param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                   "gamma": np.logspace(-2, 2, 5)})
    print('start training....')
    t0 = time.time()
    svr.fit(X_trian,train_price)
    svr_fit = time.time() - t0
    print('training time   ',svr_fit)
    joblib.dump(svr,'svr_model.pkl')
    print('saved')
    train_sizes, train_scores_svr, test_scores_svr = learning_curve(svr, X_trian, train_price,
                                                                    train_sizes=np.linspace(0.1, 1, 10),
                                                                    scoring="neg_mean_squared_error")
    plt.plot(train_sizes, -test_scores_svr.mean(1), 'o-', color="r",
             label="SVR")

    plt.xlabel("Train size")
    plt.ylabel("Mean Squared Error")
    plt.title('Learning curves')
    plt.legend(loc="best")

    plt.show()
elif TYPE=='TEST':
    model=joblib.load('svr_model.pkl')
    result=np.expm1(model.predict(X_test))
    result_ser=pd.Series(result)
    result_df=pd.concat((test_id,result_ser),axis=1)
    #result_df.index=['Id','SalePrice']
    answer=result_df.to_csv('answer.csv')
    print(result_df)