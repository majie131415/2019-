import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import model_selection, metrics
# from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

# 读取数据
train = pd.read_csv(open('train.csv', encoding='utf-8'), usecols=(0, 1, 2))
test = pd.read_csv(open('test.csv', encoding='utf-8'), usecols=(0, 1))

# 将标签列转换为数值，并删除原标签列
train['Disbursed'] = train['label'].apply(lambda x: 1 if x == 'Positive' else 0)
train = train.drop('label', axis=1)

# 词向量转换，并添加到数据集中
vectorizer = TfidfVectorizer()
xtrain = vectorizer.fit_transform(train['review'])
xtest = vectorizer.transform(test['review'])

train['reviews'] = xtrain
test['reviews'] = xtest
train = train.drop('review', axis=1)
test = test.drop('review', axis=1)

target = 'Disbursed'
IDcol = 'ID'
ytrain = train['Disbursed']

# 交叉验证
# n_folds = [x for x in range(1,6)]
# print('LogisticRegression...')

log_train = np.zeros(train.shape[0])
log_test = np.zeros(test.shape[0])

# 选定8折交叉验证
for i in range(8, 9):

    skf = StratifiedKFold(n_splits=i)

    for train_index, test_index in skf.split(xtrain, ytrain):
        log = LogisticRegression(C=5, class_weight='balanced')
        # log = SVC(probability=True)
        # log = RandomForestClassifier(n_estimators=200)
        # log = XGBClassifier(learning_rate=0.1,
        #                      n_estimators=1000,
        #                      max_depth=5,
        #                 min_child_weight=1,
        #                      gamma=0.001,
        #                      subsample=1,
        #                      colsample_bytree=0.8,
        #                      objective='binary:logistic',
        #                      nthread=4,
        #                      scale_pos_weight=1,
        #                             early_stopping_rounds=100,
        #                      seed=27)
        log.fit(xtrain[train_index], ytrain[train_index])
        train_score = log.predict_proba(xtrain[test_index])[:, 1]
        test_score = log.predict_proba(xtest)[:, 1]
        print(test_score.max())
        print('per model roc_auc_score:', metrics.roc_auc_score(ytrain[test_index], train_score))
        log_train[test_index] += train_score
        log_test += test_score
    print('model roc_auc_score:{0};n_split:{1}'.format(metrics.roc_auc_score(ytrain, log_train), i))
    #print(log_test)
    log_test =log_test/8
    #print(log_test.max())
# 结果进行标准化
# scaler = MinMaxScaler()
# log_test1 = log_test.reshape(-1, 1)
# log_test2 = scaler.fit_transform(log_test1)
# #print(log_test2.max())
#test['Pred'] = log_test
#test[['ID','Pred']].to_csv('submit1.csv',encoding='utf-8',index=False)
