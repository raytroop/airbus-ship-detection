import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


df_trn = pd.read_csv('dataset/train_ship_segmentations_v2.csv')
df_trn['numBox'] = df_trn.EncodedPixels.map(lambda x: x is not np.nan).astype(np.int)
val_count = df_trn.groupby('ImageId').numBox.sum()
val_count = pd.DataFrame(val_count)
val_count.reset_index(inplace=True)

# stratified cv
val_count['foldIdx'] = np.nan

skf = StratifiedKFold(n_splits=5, random_state=42)
for i, (trnIdx, tstIdx) in enumerate(skf.split(val_count.ImageId.values, val_count.numBox.values)):
    val_count.iloc[tstIdx, 2] = i+1

val_count['foldIdx'] = val_count.foldIdx.astype(np.int)

# map fold index
df_trn = pd.merge(df_trn[['ImageId', 'EncodedPixels']], val_count[['ImageId', 'foldIdx']], on='ImageId')
df_trn.columns = ['ImageId', 'EncodedPixels', 'foldIdx']

# write to file
if not os.path.isdir('folds'):
    os.mkdir('folds')

for i in range(5):
    val = df_trn[df_trn.foldIdx == i+1]
    trn = df_trn[df_trn.foldIdx != i+1]
    print(val.shape, trn.shape)
    val.to_csv('folds/cv5val{}.csv'.format(i+1) ,index=False, columns=['ImageId', 'EncodedPixels'])
    trn.to_csv('folds/cv5trn{}.csv'.format(i+1) ,index=False, columns=['ImageId', 'EncodedPixels'])