import torch
import numpy as np
#from maskrcnn_benchmark.data.datasets.objects import Objects_Detection_Dataset 
import argparse
#import torchmetrics.detection
#from torchmetrics.detection.mean_ap import MeanAveragePrecision
#from maskrcnn_benchmark.data.datasets.objects import Objects_Detection_Dataset 

from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from scipy import ndimage, misc
import skimage.draw as draw
import numpy as np
import pandas as pd
import torch
import os

import torchvision.datasets as dset

parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference ReadOut To Numpy")
parser.add_argument(
        "--preds_file",
        required=True,
        help="path to predictions file",
)
parser.add_argument(
        "--outputfile",
        required=True,
        help="path to outputfile",
)

args = parser.parse_args()
preds = torch.load(args.preds_file)
pred_map_list = []
for pred in preds:
    bbox = pred.bbox
    extra_fields = pred.extra_fields
    scores = extra_fields["scores"]
    labels = extra_fields["labels"]
    pred_map = dict(boxes=bbox, scores=scores, labels=labels)
    pred_map_list.append(pred_map)
np.save(args.outputfile, pred_map_list)

