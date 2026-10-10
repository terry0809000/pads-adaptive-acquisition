from pathlib import Path
import csv, argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
p=argparse.ArgumentParser();p.add_argument('--features-dir',type=Path,default=Path('pads_features_received/PADS_features_return'));args=p.parse_args()
z=np.load(args.features_dir/'PADS_motion_features.npz');ids=z['subject_ids'][z['primary_mask']];y=z['primary_y'];fold=np.full(len(y),-1)
for k,(_,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20260913).split(ids,y)):fold[test]=k
with (args.features_dir/'primary_outer_folds.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['subject_id','y','outer_fold']);w.writerows(zip(ids,y,fold))
print('Created the prespecified participant folds from the public inputs; n=',len(y))
