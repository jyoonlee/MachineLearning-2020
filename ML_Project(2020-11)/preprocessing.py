import pandas as pd
import numpy as np

def get_merged_csv(flist, **kwargs):
    return pd.concat([pd.read_csv(f, **kwargs) for f in flist], ignore_index=True)


def sampling_func(data, sample_pct):
    np.random.seed(1004)
    n = len(data)
    sample_n = sample_pct
    sample = data.take(np.random.permutation(n)[:sample_n])
    return sample


df = pd.read_csv('datasets/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv')

sample_set = df.groupby('Label').apply(sampling_func, sample_pct=50000)
sample_set.sort_index()

sample_set.to_csv('C:/Users/junso/PycharmProjects/Exercise/Sampling Result.csv')