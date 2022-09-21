from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from scipy import ndimage, misc
import skimage.draw as draw
import numpy as np
import pandas as pd
import pandas
import torch
import os

import torchvision.datasets as dset
import maskrcnn_benchmark.transforms as T
from maskrcnn_benchmark.structures.bounding_box import BoxList

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
          if "mol" in data_dir:
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
        if "clevr" in self.data_dir:
            return {"height": 320, "width": 480}
        if "mol" in self.data_dir:
            return {"height": 300, "width": 300}

      def get_transform(self, normalize=False):
              transforms = []
              transforms.append(T.ToTensor())
              return T.Compose(transforms)

      def __len__(self):
          return len([name for name in os.listdir(f"{self.data_dir}/images/")])

      def get_groundtruth(self, idx):
        if self.pseudo == 1:
            if self.fold > 0:
               try:
                  labels_df = pd.read_csv(f"{self.data_dir}/pseudo_label_{self.fold}/{str(idx)}.txt")
               except pandas.errors.EmptyDataError:
                  labels_df = pd.DataFrame()
            else:
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
        if len(boxes) == 0:
           labels.append(1) 
           boxes.append([5, 5, 15, 15])
           obj_idx+=1
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
          # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)
        difficult = torch.zeros((obj_idx-1,), dtype=torch.int64)
          #labels = torch.ones((obj_idx-1,), dtype=torch.int64)
        img_info = self.get_img_info(idx)
        image_id = torch.tensor([idx])
        if len(boxes) > 0:
           area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
           area = 0
          # suppose all instances are not crowd
        iscrowd = torch.zeros((obj_idx-1,), dtype=torch.int64)
#        print(boxes)
       # import ipdb; ipdb.sset_trace()
        if len(boxes)>0:
           target = BoxList(boxes, (img_info['width'], img_info['height']), mode="xyxy")
           target.add_field("labels", labels)
           target.add_field("difficult", difficult)
        else:
           target = None
#        print(target)
#        print(idx)
#        print(target.bbox)
#        import ipdb; ipdb.set_trace()
        #target = BoxList(boxes, (img_info['width'], img_info['height']), mode="xywh")
        #target.add_field("labels", labels)
        #target.add_field("difficult", difficult)
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
          target = target.clip_to_image(remove_empty=True)
          if self.transforms is not None:
            img, target = self.transforms(img, target)
          # print(target["image_id"])
          return img, target, idx
      
      def map_class_id_to_class_name(self, class_id):
          return Objects_Detection_Dataset.CLASSES[class_id]
