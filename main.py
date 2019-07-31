import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
import xgboost as xgb


# pandas display configuration
pd.set_option('display.float_format','{:.4f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)


def create_constant_submission():
    data_path = '/raid/data/kaggle/ieee-fraud-detection'
    test_trans = pd.read_csv(data_path + '/test_transaction.csv')
    # print(test_trans.head())
    # print(test_trans.shape)
    # print(test_trans.TransactionID.unique().shape)
    test_trans_ids = test_trans.TransactionID
    submission = pd.DataFrame(data={'TransactionID': test_trans_ids})
    submission['isFraud'] = 1.0
    print(submission.head())
    print(submission.shape)
    submission.to_csv(data_path + '/submissions/constant_submission.csv', index=False)


def csv_to_parquet(csv_path, parquet_path):
    df = pd.read_csv(csv_path)
    df.to_parquet(parquet_path)


def compute_constant_auc():
    data_root = '/raid/data/kaggle/ieee-fraud-detection'
    train_trans = pq.read_pandas(data_root + '/train_transaction.parquet').to_pandas()
    train_trans['preds'] = 1.0
    auc = roc_auc_score(train_trans.isFraud, train_trans.preds)
    print('Auc:', auc)

def get_numeric_features():
    # data_root = '/raid/data/kaggle/ieee-fraud-detection'
    # train_trans = pq.read_pandas(data_root + '/train_transaction.parquet').to_pandas()
    # cols = train_trans.columns
    # print(cols)
    dist_features = ['dist1', 'dist2']
    # print(dist_features)
    c_features = ['C' + str(i) for i in range(1, 15)]
    # print(c_features)
    d_features = ['D' + str(i) for i in range(1, 16)]
    # print(d_features)
    v_features = ['V' + str(i) for i in range(1, 340)]
    # print(v_features)
    numeric_features = dist_features + c_features + d_features + v_features
    # print(numeric_features)
    return numeric_features


def main():
    data_root = '/raid/data/kaggle/ieee-fraud-detection'

    # create_constant_submission()

    # csv_to_parquet(data_root + '/train_transaction.csv', data_root + '/train_transaction.parquet')
    # csv_to_parquet(data_root + '/test_transaction.csv', data_root + '/test_transaction.parquet')
    # csv_to_parquet(data_root + '/train_identity.csv', data_root + '/train_identity.parquet')
    # csv_to_parquet(data_root + '/test_identity.csv', data_root + '/test_identity.parquet')

    # test_trans = pq.read_pandas(data_root + '/test_transaction.parquet').to_pandas()
    # print(list(test_trans.columns))
    # train_trans = pq.read_pandas(data_root + '/train_transaction.parquet').to_pandas()
    # print(list(train_trans.columns))

    # compute_constant_auc()

    #  train_trans = pq.read_pandas(data_root + '/train_transaction.parquet').to_pandas()
    #  print(list(train_trans.columns))
    #  print(train_trans.head())
    #
    # train_identity = pq.read_pandas(data_root + '/train_identity.parquet').to_pandas()
    # print(train_identity.head())

    numeric_features = get_numeric_features()

    data = pq.read_pandas(data_root + '/train_transaction.parquet').to_pandas()

    np.random.seed(123)
    train_pct = .8
    train_mask = np.random.rand(len(data)) < train_pct
    val_mask = ~ train_mask

    train_labels = data.iloc[train_mask].isFraud
    train_features = data[numeric_features].iloc[train_mask]

    val_labels = data.iloc[val_mask].isFraud
    val_features = data[numeric_features].iloc[val_mask]

    # print(train_features.shape)
    # print(train_labels.shape)
    # print(val_features.shape)
    # print(val_labels.shape)

    xgb_params = {
        'tree_method': 'gpu_hist',
        'n_gpus': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
    }

    num_rounds = 100

    dtrain = xgb.DMatrix(train_features.values, label=train_labels.values)
    dval = xgb.DMatrix(val_features.values, label=val_labels.values)

    evals = [(dval, 'val'), (dtrain, 'train')]
    model = xgb.train(xgb_params, dtrain, evals=evals)

    test_data = pq.read_pandas(data_root + '/test_transaction.parquet').to_pandas()
    test_features = test_data[numeric_features]
    dtest = xgb.DMatrix(test_features.values)

    preds = model.predict(dtest)

    submission = pd.DataFrame(data={'TransactionID': test_data.TransactionID})
    submission['isFraud'] = preds
    print(submission.head())
    print(submission.shape)
    submission.to_csv(data_root + '/submissions/xgb_default_submission.csv', index=False)


if __name__ == '__main__':
    main()