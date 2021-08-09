#coding=gbk
from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os, sys
import pickle as pkl
import numpy as np

from args import get_parser

parser = get_parser()
opts = parser.parse_args()


def default_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im
    except:
        print(path)
        return Image.new('RGB', (224, 224), 'white')

class MyDataSet(data.Dataset):
    def __init__(self, transform=None, partition=None, loader=default_loader):
        
        self.partition = partition

        self.batch_size = opts.batch_size
        self.semantic_size = int(round(opts.semantic_pc * self.batch_size))
        self.background_size = self.batch_size - self.semantic_size
        self.same_class_num = opts.same_class_num
        self.test_lis_split = pkl.load(open(opts.test_lis_split, 'rb'))

        self.train_lis = pkl.load(open(opts.train_lis, 'rb'))
        self.valid_lis = pkl.load(open(opts.valid_lis, 'rb'))
        self.test_lis = pkl.load(open(opts.test_lis, 'rb'))

        if self.partition == 'train':
            self.ids  = self.train_lis
        elif self.partition == 'valid':
            self.ids = self.valid_lis
        elif self.partition == 'test':
            self.ids = self.test_lis
        elif self.partition == 'all':
             self.ids = self.train_lis +self.valid_lis + self.test_lis
             pkl.dump(self.ids, open('./mini_food_data/data_90/rec_ids.pkl', 'wb'))
        else:
            raise Exception('Unknown partition type %s.' % partition)

        self.classes = pkl.load(open(opts.classes, 'rb'))
        self.indices_by_class = [[] for class_id in range(opts.numClasses)]
        for index in range(len(self.ids)):
            class_id = self.classes[str(self.ids[index])]
            self.indices_by_class[class_id].append(index)
        
        self.transform = transform
        self.loader = loader

        self.titleMaxlen = opts.titleMaxlen
        self.ingrMaxlen = opts.ingrMaxlen
        self.wordMaxlen = opts.wordMaxlen
        self.imageMaxlen = opts.imageMaxlen

        self.recipe_path = opts.recipe_path
        # self.step_img_path = opts.step_img_path
        self.final_img_path = opts.final_img_path
        self.final_img_verb_path = opts.final_img_verb_path
        self.final_img_ori_path = opts.final_img_ori_path

    def zero_padding(self, feature, maxlen):
        length, width = feature.shape
        assert length <= maxlen
        if length < maxlen:
            gap = maxlen - length
            return np.concatenate([feature, np.zeros((gap, width))], axis=0)
        else:
            return feature

    def __getitem__(self, index):
        recipId = self.ids[index]

        subdir = os.path.join(self.recipe_path, str(recipId))

        title_feature = self.zero_padding(
            np.load(os.path.join(subdir, str(recipId)+'_title.npy')), self.titleMaxlen)
        ingr_feature = self.zero_padding(
            np.load(os.path.join(subdir, str(recipId)+'_ingr.npy')), self.ingrMaxlen)
        step_feature = self.zero_padding(
            np.load(os.path.join(subdir, str(recipId)+'_step.npy')), self.wordMaxlen)
        
        # stepping_feature = self.zero_padding(
        #   np.load(os.path.join(self.step_img_path, str(recipId)+'.npy')), self.imageMaxlen)


        #img_feature = np.load(os.path.join(self.final_img_path, str(recipId)+'.npy'))
        #img_feature = np.reshape(img_feature, (-1))

        img_path = os.path.join(self.final_img_ori_path, str(recipId)+'.jpg')
        img_feature = self.loader(img_path)
        if self.transform is not None:
            img_feature = self.transform(img_feature)

        #img_feature_verb = np.load(os.path.join(self.final_img_verb_path, str(recipId)+'.npy'))
        img_feature_verb = np.reshape(img_feature, (-1))

        rec_class = self.classes[str(recipId)]
        img_class = self.classes[str(recipId)]

        title_feature = np.array(title_feature, dtype='float32')
        ingr_feature = np.array(ingr_feature, dtype='float32')
        step_feature = np.array(step_feature, dtype='float32')
        #stepping_feature = np.array(stepping_feature, dtype='float32')
        img_feature_verb = np.array(img_feature_verb, dtype='float32')

        stepping_feature = np.array([], dtype='float32')

        return [title_feature, ingr_feature, step_feature, # stepping_feature,
                img_feature, img_feature_verb], [img_class, rec_class]
        

                
    def __len__(self):  
        return len(self.ids)