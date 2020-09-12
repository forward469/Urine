import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

def StratifiedGroupKFold(data, group_labels, class_labels, n_splits=5):
    data['Dataset'] = group_labels
    
    group_and_class_df = pd.DataFrame({'Dataset': group_labels, 'TOTAL_a': class_labels})
    GnC_df_shortened = group_and_class_df.drop_duplicates(subset='Dataset').reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_splits)
    i = 0
    GnC_train, GnC_test = list(range(n_splits)), list(range(n_splits))
    train_data, test_data = list(range(n_splits)), list(range(n_splits))

    for train_index, test_index in skf.split(GnC_df_shortened['Dataset'], GnC_df_shortened['TOTAL_a']):
        GnC_train[i], GnC_test[i] = GnC_df_shortened.iloc[train_index, : ], GnC_df_shortened.iloc[test_index, : ]
        train_data[i] = pd.merge(left=data, right=GnC_train[i], how='right', on='Dataset')
        test_data[i] = pd.merge(left=data, right=GnC_test[i], how='right', on='Dataset')
        i += 1

    
    return train_data, test_data