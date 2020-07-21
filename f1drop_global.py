from __future__ import print_function
import numpy as np
np.random.seed(1)
import pandas as pd
import deepmatcher as dm

# generate data for training deepmatcher
datadir = 'datasets/Structured/Walmart-Amazon'

columns_to_shuffle = []

test_df = pd.read_csv(datadir + '/merged_test.csv', encoding='utf-8', sep=",")
test_df2 = pd.read_csv(datadir + '/merged_test.csv', encoding='utf-8', sep=",")
test_df2 = test_df2.sample(frac=1) # shuffle
test_df2.reset_index(inplace=True, drop=True)
for c in test_df2.columns:
    if c == 'label' or c == 'id' or ((c.startswith('ltable_') or (c.startswith('rtable_'))) and c[7:] not in columns_to_shuffle):
        test_df2[c] = test_df[c]

test_df2.to_csv(datadir+'/merged_test_shuffle.csv', index=False )

trainLab, validationLab, testLab = dm.data.process(path=datadir, left_prefix='ltable_',right_prefix='rtable_',
                                                   train='merged_train.csv',  validation='merged_valid.csv',
                                                   test='merged_test_shuffle.csv')

# train deepmatcher
model = dm.MatchingModel(attr_summarizer='hybrid')
model.load_state('wa_dm.pth')

# evaluate deepmatcher on test data
stats = model.run_eval(testLab)
print(f'final f1 :{stats.f1()}')
