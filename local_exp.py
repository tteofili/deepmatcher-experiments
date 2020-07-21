import pandas as pd
import math, re, os, random, string
from collections import Counter
import numpy as np
import deepmatcher as dm
from utils.deepmatcher_utils import wrapDm
import os, contextlib, random, string
from erexplain import explainSamples

WORD = re.compile(r'\w+')


# calculate similarity between two text vectors
def get_cosine(text1, text2):
    vec1 = text_to_vector(text1)
    vec2 = text_to_vector(text2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


def find_candidates(record, source, min_similarity, find_positives):
    record2text = " ".join([str(val) for k, val in record.to_dict().items() if k not in ['id']])
    source_without_id = source.copy()
    source_without_id = source_without_id.drop(['id'], axis=1)
    source_ids = source.id.values
    # for a faster iteration
    source_without_id = source_without_id.values
    candidates = []
    for idx, row in enumerate(source_without_id):
        currentRecord = " ".join(str(row))
        currentSimilarity = get_cosine(record2text, currentRecord)
        if find_positives:
            if currentSimilarity >= min_similarity:
                candidates.append((record['id'], source_ids[idx]))
        else:
            if currentSimilarity < min_similarity:
                candidates.append((record['id'], source_ids[idx]))
    return pd.DataFrame(candidates, columns=['ltable_id', 'rtable_id'])


def __generate_unlabeled(dataset_dir, unlabeled_filename, lprefix='ltable_', rprefix='rtable_'):
    df_tableA = pd.read_csv(os.path.join(dataset_dir, 'tableA.csv'), dtype=str)
    df_tableB = pd.read_csv(os.path.join(dataset_dir, 'tableB.csv'), dtype=str)
    unlabeled_ids = pd.read_csv(os.path.join(dataset_dir, unlabeled_filename), dtype=str)
    unlabeled_ids.columns = ['id1', 'id2']
    left_columns = list(map(lambda s: lprefix + s, list(df_tableA)))
    right_columns = list(map(lambda s: rprefix + s, list(df_tableB)))
    df_tableA.columns = left_columns
    df_tableB.columns = right_columns

    unlabeled_df = unlabeled_ids.merge(df_tableA, how='inner', left_on='id1', \
                                       right_on=lprefix + 'id').merge(df_tableB, how='inner', left_on='id2',
                                                                      right_on=rprefix + 'id')
    unlabeled_df[lprefix + 'id'] = unlabeled_df[lprefix + 'id'].astype(str)
    unlabeled_df[rprefix + 'id'] = unlabeled_df[rprefix + 'id'].astype(str)
    unlabeled_df['id'] = "0@" + unlabeled_df[lprefix + 'id'] + "#" + "1@" + unlabeled_df[rprefix + 'id']
    unlabeled_df = unlabeled_df.drop(['id1', 'id2', lprefix + 'id', rprefix + 'id'], axis=1)
    return unlabeled_df


def get_original_prediction_deepmatcher(r1, r2, model, lprefix='ltable_', rprefix='rtable_'):
    r1_df = pd.DataFrame(data=[r1.values], columns=r1.index)
    r2_df = pd.DataFrame(data=[r2.values], columns=r2.index)
    r1_df.columns = list(map(lambda col: lprefix + col, r1_df.columns))
    r2_df.columns = list(map(lambda col: rprefix + col, r2_df.columns))
    r1r2 = pd.concat([r1_df, r2_df], axis=1)
    r1r2['id'] = "0@" + str(r1r2[lprefix + 'id'].values[0]) + "#" + "1@" + str(r1r2[rprefix + 'id'].values[0])
    r1r2 = r1r2.drop([lprefix + 'id', rprefix + 'id'], axis=1)
    prediction = wrapDm(r1r2, model)
    return prediction, r1r2


def create_dataset_4local_explanation(r1, r2, model, lsource, rsource, dataset_dir,
                                      min_similarity_match, max_similarity_non_match,
                                      predict_fn, num_triangles=100):
    originalPrediction, r1r2 = get_original_prediction_deepmatcher(r1, r2, model)
    if originalPrediction[0] > originalPrediction[1]:
        findPositives = True
        candidates4r1 = find_candidates(r1, rsource, min_similarity_match, find_positives=findPositives)
        candidates4r2 = find_candidates(r2, lsource, min_similarity_match, find_positives=findPositives)
    else:
        findPositives = False
        candidates4r1 = find_candidates(r1, rsource, max_similarity_non_match, find_positives=findPositives)
        candidates4r2 = find_candidates(r2, lsource, max_similarity_non_match, find_positives=findPositives)
    id4explanation = pd.concat([candidates4r1, candidates4r2], ignore_index=True)
    tmp_name = "./{}.csv".format("".join([random.choice(string.ascii_lowercase) for _ in range(10)]))
    id4explanation.to_csv(os.path.join(dataset_dir, tmp_name), index=False)
    unlabeled_df = __generate_unlabeled(dataset_dir, tmp_name)
    os.remove(os.path.join(dataset_dir, tmp_name))
    unlabeled_predictions = predict_fn(unlabeled_df, model, outputAttributes=True)
    if findPositives:
        neighborhood = unlabeled_predictions[unlabeled_predictions.match_score >= 0.5].copy()
    else:
        neighborhood = unlabeled_predictions[unlabeled_predictions.match_score < 0.5].copy()
    if len(neighborhood) > num_triangles:
        neighborhood = neighborhood.sample(n=num_triangles)
    neighborhood['id'] = neighborhood.index
    neighborhood['label'] = list(map(lambda predictions: int(round(predictions)),
                                     neighborhood.match_score.values))
    neighborhood = neighborhood.drop(['match_score'], axis=1)
    r1r2['label'] = np.argmax(originalPrediction)
    dataset4explanation = pd.concat([r1r2, neighborhood], ignore_index=True)
    return dataset4explanation


lsource = pd.read_csv('datasets/Structured/DBLP-ACM/tableA.csv')
rsource = pd.read_csv('datasets/Structured/DBLP-ACM/tableB.csv')

model = dm.MatchingModel(attr_summarizer='hybrid')
model.load_state('da_dm.pth')


def predict_fn(test_df, model, ignore_columns=['label'], outputAttributes=False, batch_size=32):
    data = test_df.copy().drop([c for c in ignore_columns if c in test_df.columns], axis=1)
    if not ('id' in data.columns):
        data['id'] = np.arange(len(data))
    tmp_name = "./{}.csv".format("".join([random.choice(string.ascii_lowercase) for _ in range(10)]))
    data.to_csv(tmp_name, index=False)
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            data_processed = dm.data.process_unlabeled(tmp_name, trained_model=model)
            predictions = model.run_prediction(data_processed, output_attributes=outputAttributes, \
                                               batch_size=batch_size)
            out_proba = predictions['match_score'].values
    multi_proba = np.dstack((1 - out_proba, out_proba)).squeeze()
    os.remove(tmp_name)
    if outputAttributes:
        return predictions
    else:
        return multi_proba


for l_idx in range(len(lsource)):
    for r_idx in range(len(rsource)):
        try:
            l_tuple = lsource.iloc[l_idx]
            r_tuple = rsource.iloc[r_idx]
            dataset4local = create_dataset_4local_explanation(l_tuple, r_tuple, model, lsource,
                                                              rsource, 'datasets/Structured/DBLP-ACM/', 0.01, -0.5,
                                                              predict_fn)
            c2ex = np.argmax(get_original_prediction_deepmatcher(l_tuple, r_tuple, model)[0])
            if len(dataset4local) > 2:
                explanation, flipped_pred = explainSamples(dataset4local, [lsource, rsource], model, predict_fn,
                                                           class_to_explain=c2ex, maxLenAttributeSet=3)
                print(explanation)
        except ValueError as e:
            pass
