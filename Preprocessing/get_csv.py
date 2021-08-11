
#import pandas as pd
#tsv_file='../work/train.text.tsv'
#csv_table=pd.read_table(tsv_file,sep='\t')
#csv_table.to_csv('../work/data.csv',index=False)

import readability
import pandas as pd
import re
import numpy as np
from scipy import stats
from collections import Counter

def normalization(df):
    x = np.nan_to_num(stats.zscore(df))
    return x

def extract_readability_features(text):
    text = re.sub(r'\.', '.\n', text)
    text = re.sub(r'\?', '?\n', text)
    text = re.sub(r'!', '!\n', text)
    features = dict(readability.getmeasures(text, lang='en'))
    result = {}
    for d in features:
        result.update(features[d])
    del result['paragraphs']
    result = pd.Series(result)
    return result

def extract_NRC_features(x, sentic_df):
    # tokens = re.sub('[^a-zA-Z]', ' ', x).split()
    tokens = x.split()
    tokens = Counter(tokens)
    df = pd.DataFrame.from_dict(tokens, orient='index', columns=['count'])
    merged_df = pd.merge(df, sentic_df, left_index=True, right_index=True)
    for col in merged_df.columns[1:]:
        merged_df[col] *= merged_df["count"]
    result = merged_df.sum()
    result /= result["count"]
    result = result.iloc[1:]
    return result

text = '''This demo tool lets you enter your own text and sample some of the languages and voices that we offer.
Please note:  Not all languages and voices are available for every solution. Also, more voices are available for certain solutions. See our Languages & Voices page for a complete list of available languages for each solution.'''
res=[]
NRC_path1 = r'C:\Users\Srinu\Desktop\pers pred\meta_features_data/NRC-VAD-Lexicon.txt'
NRC_df1 = pd.read_csv(NRC_path1, index_col=['Word'], sep='\t')
tmp = extract_readability_features(text)
print(list(tmp.index))
ind = list(tmp.index)
p = normalization(tmp)
print(p,type(p))
#result = pd.concat[tmp], axis=1)
#output_file = op_dir + dataset_type + '_readability.csv'
#result.to_csv(output_file, index=False)
result = pd.DataFrame(columns = ind)
df_length = len(result)
result.loc[df_length] = p
#result = res.transpose()
result.insert(0, "user", [1], True)
#output_file = '/content/drive/MyDrive/Colab Notebooks/readability1.csv'
#result.to_csv(output_file,index=False)
tmp = extract_NRC_features(text, NRC_df1)
print(list(tmp))
res.extend(p)
res.extend(list(tmp))
print(res,len(res))

#output_file = '/content/drive/MyDrive/Colab Notebooks/nrc1.csv'
#result.to_csv(output_file,index=False)