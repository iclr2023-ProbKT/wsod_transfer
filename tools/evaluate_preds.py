import torch
import numpy as np
#from maskrcnn_benchmark.data.datasets.objects import Objects_Detection_Dataset 
import argparse
#import torchmetrics.detection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
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

class Objects_Detection_Dataset(Dataset):
       
      CLASSES = (
        "__background__ ",
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
      )

      def __init__(self, data_dir, transforms=None, box_loss_mask=False, rgb=True, pseudo = 0, fold = 0, re_train = False, in_memory = False):
          self.data_dir = data_dir
          self.pseudo   = pseudo
      #    transforms = []
      #    transforms.append(T.ToTensor())
                        #T.Compose(transforms)
      #    self.transforms = T.Compose(transforms)
          self.transforms = transforms
          self.re_train = re_train
          self.box_loss_mask = box_loss_mask
          self.fold     = fold

          self.rgb = rgb
          if in_memory: # images are stored in memory rather than having to IO. Currently not much speed up observed there
            self.images_dict = {f: Image.open(os.path.join(self.data_dir,"images",f)).convert("L") for f in os.listdir(f"{self.data_dir}/images/")}
            if self.re_train:
                self.labels_dict = {f:pd.read_csv(f"{self.data_dir}/re_labels/{f}") for f in os.listdir(f"{self.data_dir}/re_labels/")}
            else:
                self.labels_dict = {f:pd.read_csv(f"{self.data_dir}/labels/{f}") for f in os.listdir(f"{self.data_dir}/labels/")}

          self.in_memory = in_memory
          self.ids = range(self.__len__())
          if "molecules" in data_dir:
              self.label_shift = 1
          elif "mnist" in data_dir: # shifting the label index by 1 in the mnist case (0 is background in RCNN)
              self.label_shift = 1
          elif "clevr" in data_dir: # shifting the label index by 1 in the mnist case (0 is background in RCNN)
              self.label_shift = 1
         # train_idx = np.load(os.path.join(DATA_DIR,f"{self.data_dir}","folds",str(self.fold),"train_idx.npy"))
          #           import ipdb; ipdb.set_trace()
         # self.train = Subset(dataset,train_idx)

      
      def get_img_info(self, index):
        #img_id = self.ids[index]
        #anno = ET.parse(self._annopath % img_id).getroot()
        #size = anno.find("size")
        #im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": 320, "width": 480}

      def get_transform(self, normalize=False):
              transforms = []
              transforms.append(T.ToTensor())
              return T.Compose(transforms)

      def __len__(self):
          return len([name for name in os.listdir(f"{self.data_dir}/images/")])

      def get_groundtruth(self, idx):
        if self.pseudo == 1:
            labels_df = pd.read_csv(f"{self.data_dir}/pseudo_label/{str(idx)}.txt")
        else:
            labels_df = pd.read_csv(f"{self.data_dir}/labels/{str(idx)}.txt")
        labels = []
        boxes = []
        obj_idx = 1
        for index, row in labels_df.iterrows():
              labels.append(int(row['label'])+self.label_shift)
              xmin = int(row['xmin'])
              xmax = int(row['xmax'])
              ymin = int(row['ymin'])
              ymax = int(row['ymax'])
              if self.box_loss_mask:
                 boxes.append([0, 0, 0, 0])
              else:
                 boxes.append([xmin, ymin, xmax, ymax])
              start = (xmin,ymin)
              end = (xmax,ymax)
              obj_idx+=1
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
          # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)
        difficult = torch.zeros((obj_idx-1,), dtype=torch.int64)
          #labels = torch.ones((obj_idx-1,), dtype=torch.int64)
        img_info = self.get_img_info(idx)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
          # suppose all instances are not crowd
        iscrowd = torch.zeros((obj_idx-1,), dtype=torch.int64)
#        print(boxes)
        #import ipdb; ipdb.set_trace()
        target = {}
        target['boxes']=boxes
        #target = BoxList(boxes, (img_info['width'], img_info['height']), mode="xyxy")
#        print(target)
#        print(idx)
#        print(target.bbox)
#        import ipdb; ipdb.set_trace()
        #target = BoxList(boxes, (img_info['width'], img_info['height']), mode="xywh")
        #target.add_field("labels", labels)
        #target.add_field("difficult", difficult)
        target['labels']=labels
        return target

      def __getitem__(self, idx):
          imagename=f"{self.data_dir}/images/{str(idx)}.png"
          if self.in_memory:
            img = self.images_dict[f"{str(idx)}.png"]
            labels_df = self.labels_dict[f"{str(idx)}.txt"]
          else:
            img = Image.open(imagename).convert("L")
            if self.rgb:
                img = img.convert('RGB')
          target = self.get_groundtruth(idx)
          #target = target.clip_to_image(remove_empty=True)
          if self.transforms is not None:
            img, target = self.transforms(img, target)
          # print(target["image_id"])
          return img, target, idx
      
      def map_class_id_to_class_name(self, class_id):
          return Objects_Detection_Dataset.CLASSES[class_id]

parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference Evaluation")
parser.add_argument(
        "--preds_file",
        required=True,
        help="path to predictions file",
)
parser.add_argument(
        "--true_data_dir",
        required=True,
        help="path to true labels/dataset",
)

args = parser.parse_args()
#dataset = Objects_Detection_Dataset(data_dir="/home/moldenho/Projects/rcnn-deepproblog/generate_data/clevr/clevr_all/train/")
dataset = Objects_Detection_Dataset(data_dir=args.true_data_dir)
##preds = torch.load(args.preds_file)
pred_map_list = np.load(args.preds_file, allow_pickle=True)
Y = []
Y_hat = []
map_metric = MeanAveragePrecision()
#for pred,sample in zip(preds,dataset):
##pred_map_list = []
for pred_map,sample in zip(pred_map_list,dataset):
   # print(pred)
    Y.append(sample[1]['labels'])
    Y_hat.append(pred_map['labels'])
    target_map = dict(boxes=sample[1]['boxes'], labels=sample[1]['labels'])
    #import ipdb; ipdb.set_trace()
    map_metric.update([pred_map],[target_map])
    #import ipdb; ipdb.set_trace()
##for pred in preds:
##    bbox = pred.bbox
##    extra_fields = pred.extra_fields
##    scores = extra_fields["scores"]
##    labels = extra_fields["labels"]
##    pred_map = dict(boxes=bbox, scores=scores, labels=labels)
##    pred_map_list.append(pred_map)
    #import ipdb; ipdb.set_trace()
    #target_map = dict(boxes=sample[1].bbox.numpy(), labels=sample[1].extra_fields['labels'].numpy())
    #map_metric.update(pred_map,target_map)
    #import ipdb; ipdb.set_trace()
##np.save("pred_map_list", pred_map_list)
#import ipdb; ipdb.set_trace()
accuracy = np.array([torch.equal(Y[i].sort()[0],Y_hat[i].sort()[0]) for i in range(len(Y))]).mean()
mAP = map_metric.compute()
print(accuracy)
print(mAP)

