#!/usr/bin/env python
'''
NOTE: This uses the alannlp PyTorch implementation of Elmo!
Process a line corpus and convert text to elmo embeddings, save as json array of sentence vectors.
This expects the sents corpus which has fields title, article, domains. and treates
title as the first sentence, then splits the article sentences. The domains are ignored
'''

import argparse
import os
import json
import math
import torch
from transformers import BertTokenizer, BertModel
import readability
import re
import numpy as np
from scipy import stats
import pandas as pd
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

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

configs = {
    "small": "elmo_2x1024_128_2048cnn_1xhighway_options.json",
    "medium": "elmo_2x2048_256_2048cnn_1xhighway_options.json",
    "original5b": "elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
    "original": "elmo_2x4096_512_2048cnn_2xhighway_options.json"
}

models = {
    "small": "elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5",
    "medium": "elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5",
    "original5b": "elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
    "original": "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

}
n_hl = 12
hidden_dim = 768
MODEL = (BertModel, BertTokenizer, 'bert-base-uncased')
model_class, tokenizer_class, pretrained_weights = MODEL
model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)  # output_attentions=False
tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)
NRC_path1 = "data/NRC-Emotion-Lexicon.xlsx"
NRC_df1 = pd.read_excel(NRC_path1, index_col=0)
NRC_path2 = "data/NRC-VAD-Lexicon.txt"
NRC_df2 = pd.read_csv(NRC_path2, index_col=['Word'], sep='\t')

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print('GPU found (', torch.cuda.get_device_name(torch.cuda.current_device()), ')')
    torch.cuda.set_device(torch.cuda.current_device())
    print('num device avail: ', torch.cuda.device_count())

else:
    DEVICE = torch.device('cpu')
    print('running on cpu')

if __name__ == '__main__':

    default_m = "original"
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Input file, should be in sent1 format")
    parser.add_argument("outfile", type=str, help="Output file, contains standard cols 0.3, plus json vectors")
    parser.add_argument("-b", type=int, default=50, help="Batchsize (50)")
    parser.add_argument("-l", type=int, default=1000, help="Log every (1000)")
    parser.add_argument("--maxtoks", type=int, default=200, help="Maximum number of tokens per sentence to use (200)")
    parser.add_argument("--maxsents", type=int, default=200,
                        help="Maximum number of sentences per article to use (200)")
    parser.add_argument("-m", type=str, default=default_m,
                        help="Model (small, medium, original, original5b ({})".format(default_m))
    parser.add_argument("-g", action='store_true', help="Use the GPU (default: don't)")
    parser.add_argument("--concat", action='store_true', help="Concatenate representations instead of averaging")
    args = parser.parse_args()

    outfile = args.outfile
    infile = args.infile
    batchsize = args.b
    every = args.l
    # use_gpu = args.g
    # model = os.path.join("elmo", models[args.m])
    config = os.path.join("elmo", configs[args.m])
    concat = args.concat
    maxtoks = args.maxtoks
    maxsents = args.maxsents

    print("Loading model {}...".format(args.m))

    # elmo = ElmoEmbedder(options_file=config, weight_file=model, cuda_device=device)
    # elmo = ElmoTokenEmbedder(options_file=config, weight_file=model)
    token_len = []
    input_id = []
    token_length = 50

    print("Processing lines...")
    with open(infile, "rt", encoding="utf8") as inp:
        nlines = 0
        with open(outfile, "wt", encoding="utf8") as outp:
            for line in inp:
                #print('new line')
                fields = line.split("\t")
                title = fields[5]
                tmp = fields[4]
                tmp = tmp.split(" <splt> ")[:maxsents]
                sents = [title]
                sents.extend(tmp)
                # now processes the sents in batches
                outs = []
                # unlike the tensorflow version we can have dynamic batch sizes here!
                print(f'line no:{fields[0]},sents len:{len(sents)}')
                print(math.ceil(len(sents) / batchsize))
                for batchnr in range(math.ceil(len(sents) / batchsize)):
                    print("next batch")
                    avg = []
                    fromidx = batchnr * batchsize
                    toidx = (batchnr + 1) * batchsize
                    actualtoidx = min(len(sents), toidx)
                    # print("Batch: from=",fromidx,"toidx=",toidx,"actualtoidx=",actualtoidx)
                    sentsbat = sents[fromidx:actualtoidx]
                    # print("len of sentsbat:",len(sentsbat))
                    # print("len of sentsbat:",len(sentsbat[0]))
                    sentsbatch = [(" ").join(s.split()[:maxtoks]) for s in sentsbat]
                    # print("len of sentsbatch:",len(sentsbatch))
                    # print("len of sentsbatch:",len(sentsbatch[0]))

                    print('sensbatch len:', len(sentsbatch))
                    for s in sentsbatch:
                        if len(s) == 0:
                            s.append("")  # otherwise we get a shape (3,0,dims) result
                    #print("checkpoint0")

                    for sentence in sentsbatch:
                        # q=[]
                        input_id = []
                        #print(sentence)
                        tokens = tokenizer.tokenize(sentence)
                        #print("checkpoint 0.1")

                        token_len.append(len(tokens))
                        token_ids = tokenizer.encode(tokens, add_special_tokens=True, max_length=token_length,
                                                     pad_to_max_length=True)
                        input_id.append(token_ids)
                        input_ids = torch.from_numpy(np.array(input_id)).long()
                        #print("checkpoint 1")
                        # print(input_ids)
                        #print("checkpoint 1.1")
                        bert_output = model(input_ids)
                        tmph = []
                        for ii in range(n_hl):
                            tmph.append((bert_output[2][ii + 1].detach().numpy()).mean(axis=1))
                        # hidden_features.append(np.array(tmph))
                        alphaW = np.full([n_hl], 1 / n_hl)
                        hidden_features = np.einsum('k,kij->ij', alphaW, tmph)
                        #print(hidden_features.shape)
                        #print(hidden_features[0].shape)

                        #try:
                          #read = extract_readability_features(sentence)
                          #read_feat = normalization(read)
                        #except ValueError:
                          #print(f'value Error with {sentence}')
                          #read_feat = [0]*31
                        
                        #emotion = extract_NRC_features(sentence, NRC_df1)
                        #vad = extract_NRC_features(sentence, NRC_df2)
                        
                        #feat = np.append(hidden_features[0],read_feat)
                        #feat = np.append(feat,list(emotion))
                        #feat = np.append(feat,list(vad))

                        #print(len(hidden_features), len(hidden_features[0]))
                        #print(hidden_features[0].shape,feat.shape)
                        #print("checkpoint 2")

                        avg.append(hidden_features[0])
                      
                    #print("checkpoint3")
                    outs.extend(avg)
                    print(len(outs))

                # print("Result lines:", len(outs))
                outs = [a.tolist() for a in outs]
                if not outs:
                    print('empty')

                    # outs=[-0.02828037366271019, -0.02817110197308163, -0.02776097149277727, -0.028012847527861595, -0.028245604364201427, -0.02613477672760685, -0.027191620552912354, -0.026653081954767305, -0.027391517146800954, -0.027306203187132876]
                print(fields[0], fields[1], fields[2], fields[3], json.dumps(outs), sep="\t", file=outp)
                print('line done')
                nlines += 1
                if nlines % every == 0:
                    print("Processed lines:", nlines)
    print("Total processed lines")
