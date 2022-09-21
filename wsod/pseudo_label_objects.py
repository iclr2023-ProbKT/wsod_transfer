from tqdm import tqdm
import os, sys
import pandas as pd
import json
import torch, numpy as np
from maskrcnn_benchmark.data.build import build_dataset
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.config import cfg
paths_catalog = import_file(
    "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
)
DatasetCatalog = paths_catalog.DatasetCatalog

local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
if local_rank != 0:
    sys.exit()

folder, tag, it, th = sys.argv[1:]
print ('pseudo label params:', folder, tag, it, th)

###############################################################################
# VOC 2007 trainval pseudo labeling

# output label file names
#json_voc = 'voc_2007_trainval_%s_it%s_%s.json' % (tag, it, th)

path = folder + '/predictions.pth'
print ("read", path)
d_trainval = torch.load(path)
if "mnist" in folder:
   dataset_list = ("mnist_all",)
if "clevr" in folder:
   dataset_list = ("clevr_all_nofold",)
if "mol" in folder:
   dataset_list = ("mol_all_nofold",)
transforms = None
datasets = build_dataset(dataset_list, transforms, DatasetCatalog, False)

gt_trainval = [datasets[0].get_groundtruth(idx) for idx in range(len(datasets[0]))]

p_trainval = {}
p_trainval.update(zip(datasets[0].ids, zip(gt_trainval, d_trainval)))


#with open('datasets/voc/VOC2007/voc_2007_trainval.json','r') as f:
#    d = json.load(f)

th = float(th)
annos = []
id = 0
for img_id, (t, p) in tqdm(p_trainval.items(), mininterval=20):
    df = pd.DataFrame()
    img_labels = set(t.get_field('labels').tolist())
    p = p.resize(t.size)

    boxes = p.bbox.cpu().numpy()
    scores = p.get_field('scores').tolist()
    labels = p.get_field('labels').tolist()
    sortidx = np.argsort(scores)[::-1]
    labels_hit = set()
    for i in sortidx:
        l = labels[i]
        if l in img_labels and (scores[i] > th or l not in labels_hit):
            labels_hit.add(l)
            bbox = boxes[i].copy()
            bbox_copy = boxes[i].copy()
            bbox[2:] -= bbox[:2] - 1
            bbox = bbox.tolist()
            id += 1
            df = df.append({'label': int(l)-1, 'xmin': int(bbox_copy[0]), 'ymin': int(bbox_copy[1]), 'xmax': int(bbox_copy[2]), 'ymax': int(bbox_copy[3])}, ignore_index=True)
    if "mnist" in folder:
       df.astype(int).to_csv(os.path.join('/home/moldenho/Projects/ProbKT/generate_data/mnist/mnist3_all/train/','pseudo_label',f"{img_id}.txt"), index = False)
    if "clevr" in folder:
       df.astype(int).to_csv(os.path.join('../rcnn-deepproblog/generate_data/clevr/clevr_all/train/','pseudo_label',f"{img_id}.txt"), index = False)
    if "mol" in folder:
       df.astype(int).to_csv(os.path.join('../rcnn-deepproblog/generate_data/molecules/molecules_all/train/','pseudo_label',f"{img_id}.txt"), index = False)
