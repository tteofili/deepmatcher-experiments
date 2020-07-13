from __future__ import print_function
import numpy as np
np.random.seed(1)
import pandas as pd
import deepmatcher as dm
import spacy
from collections import defaultdict
from utils.deepmatcher_utils import wrapDm
from anchor import anchor_text
from collections import Counter


# utility functions to perform predictions with deepmatcher based on anchors requirements
def makeAttr(attribute,idx,isLeft):
    attr_prefixed = []
    for token in attribute.split():
        if isLeft:
            attr_prefixed.append('L'+str(idx)+'_'+token)
        else:
            attr_prefixed.append('R'+str(idx)+'_'+token)
    return " ".join(attr_prefixed)

def pairs_to_string(df,lprefix,rprefix,ignore_columns = ['id','label']):
    pairs_string = []
    l_columns = [col for col in list(df) if (col.startswith(lprefix)) and (col not in ignore_columns)]
    r_columns = [col for col in list(df) if col.startswith(rprefix) and (col not in ignore_columns)]
    df = df.fillna("")
    for i in range(len(df)):
        this_row = df.iloc[i]
        this_row_str = []
        for j,lattr in enumerate(l_columns):
            this_attr = makeAttr(str(this_row[lattr]),j,isLeft=True)
            this_row_str.append(this_attr)
        for k,rattr in enumerate(r_columns):
            this_attr = makeAttr(str(this_row[rattr]),k,isLeft=False)
            this_row_str.append(this_attr)
        pairs_string.append(" ".join(this_row_str))
    return pairs_string

def makeRow(pair_str,attributes,lprefix,rprefix):
    row_map = defaultdict(list)
    for token in pair_str.split():
        if token.startswith('L') or token.startswith('R'):
            row_map[token[:2]].append(token[3:])
    row = {}
    for key in row_map.keys():
        if key.startswith('L'):
            ## key[1] is the index of attribute
            this_attr = lprefix+attributes[int(key[1])]
            row[this_attr] = " ".join(row_map[key])
        else:
            this_attr = rprefix+attributes[int(key[1])]
            row[this_attr] = " ".join(row_map[key])
    keys = dict.fromkeys(row.keys(),[])
    for r in keys: # add any completely missing attribute (with '' value)
        if r.startswith(lprefix):
            twin_attr = 'r'+r[1:]
            if None == row.get(twin_attr):
                row[twin_attr] = ''
        elif r.startswith(rprefix):
            twin_attr = 'l' + r[1:]
            if None == row.get(twin_attr):
                row[twin_attr] = ''
    return pd.Series(row)

def pairs_str_to_df(pairs_str_l,columns,lprefix,rprefix):
    lschema = list(filter(lambda x: x.startswith(lprefix),columns))
    schema = {}
    for i, s in enumerate(lschema):
        schema[i] = s.replace(lprefix, "")
    allTuples = []
    for pair_str in pairs_str_l:
        row = makeRow(pair_str,schema,'ltable_','rtable_')
        allTuples.append(row)
    df = pd.DataFrame(allTuples)
    df['id'] = np.arange(len(df))
    return df

def pair_str_to_df(pair_str,columns,lprefix,rprefix):
    lschema = list(filter(lambda x: x.startswith(lprefix),columns))
    schema = {}
    for i, s in enumerate(lschema):
        schema[i] = s.replace(lprefix, "")
    row = makeRow(pair_str,schema,'ltable_','rtable_')
    row['id'] = 0
    return pd.DataFrame( data = [row.values],columns= row.index)

def wrap_dm(model, stringTuples):
    df = pairs_str_to_df(stringTuples, test_df.columns,'ltable_','rtable_')
    df['id'] = np.arange(len(df))
    predictions = wrapDm(df, model)
    if predictions.shape==(2,):
        return np.array([np.argmax(predictions)])
    else:
        return np.argmax(predictions,axis=1)


predict_fn = lambda tuples : wrap_dm(model, tuples)

# generate data for training deepmatcher
datadir = 'datasets/Structured/DBLP-ACM'
trainLab, validationLab, testLab = dm.data.process(path=datadir, left_prefix='ltable_',right_prefix='rtable_',
                                                   train='merged_train.csv',  validation='merged_validation.csv',
                                                   test='merged_test.csv')

# train deepmatcher
model = dm.MatchingModel(attr_summarizer='hybrid')
model.load_state('da_dm.pth')
#model.run_train(trainLab, validationLab, best_save_path='da_dm.pth', epochs=15)

# evaluate deepmatcher on test data
eval = model.run_eval(testLab)

# transform test data to feed it to anchors
test_df = pd.read_csv(datadir + '/merged_test.csv')
pairs_str_test = pairs_to_string(test_df,'ltable_','rtable_')

# create anchors text explainer instance
class_names = ["non-matching","matching"]
nlp = spacy.load('en_core_web_lg')
explainer = anchor_text.AnchorText(nlp, class_names, mask_string='', use_unk_distribution=True, use_bert=False)

verbose = False
e_values = {0: [''], 1: ['']}
threshold = 51
print(f'using {len(pairs_str_test)} test samples')
for t_i in pairs_str_test:
    try:
        if len(e_values[0]) >= threshold and len(e_values[1]) >= threshold:
            print('finished!')
            break
        # perform prediction on test instance
        fn_result = predict_fn([t_i])
        result_key = fn_result[0]
        if len(e_values[result_key]) < threshold:
            pred = explainer.class_names[result_key]
            alternative =  explainer.class_names[1 - result_key]
            print('Prediction: %s' % pred)

            # explain test instance
            exp = explainer.explain_instance(t_i, predict_fn, threshold=0.95, use_proba=True, beam_size=2)

            # print output explanation
            print('Anchor: %s' % (' AND '.join(exp.names())))
            pev = set()
            for en in exp.names():
                pev.add(en[:2])

            e_values[result_key].append(pev)
            if verbose:
                print('Precision: %.2f' % exp.precision())
                print()
                print('Examples where anchor applies and model predicts %s:' % pred)
                print()
                print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
                print()
                print('Examples where anchor applies and model predicts %s:' % alternative)
                print()
                print('\n'.join([x[0] for x in exp.examples(partial_index=0, only_different_prediction=True)]))
    except ValueError as e:
        print(f'could not get anchor: {e}')
    except AssertionError as ae:
        print(f'failed assertion: {ae}')

for e in e_values.keys():
    vv = e_values.get(e)
    l = [frozenset(i) for i in vv]
    counter = Counter(l)
    print(counter.most_common())
