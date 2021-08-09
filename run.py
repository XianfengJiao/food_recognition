#!/usr/bin/python
#-*- coding: utf-8 -*-

import os
import time
import random
import numpy as np
import pickle as pkl


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn

from batch_sampler import BatchSamplerTriplet
from data_loader import MyDataSet
from loss import Triplet
from args import get_parser
from model import full_model


parser = get_parser()
opts = parser.parse_args()

device = torch.device('cuda')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # only one cuda device
torch.cuda.empty_cache()

def setup_seed(seed):
     torch.manual_seed(seed) # Sets the seed for generating random numbers. Returns a torch.Generator object.
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def main():
    model = full_model(titleDim = opts.titleDim,
                        ingrDim = opts.ingrDim,
                        wordDim = opts.wordDim,
                        wordModelDim = opts.wordModelDim,
                        imageDim = opts.imageDim,
                        titleMaxlen = opts.titleMaxlen,
                        ingrMaxlen = opts.ingrMaxlen,
                        wordMaxlen = opts.wordMaxlen,
                        imageMaxlen = opts.imageMaxlen,
                        margin = opts.margin,
                        num_layer = opts.numLayer,
                        num_heads = opts.numHeads,
                        embedding_dim = opts.embDim,
                        dropout_rate = opts.dropout).to(device=device)
    # model.cuda(device=device)

    criterion = Triplet().cuda(device=device)

    img_params = list(map(id, model.image.resnet.parameters()))
    rec_params = filter(lambda p: id(p) not in img_params, model.parameters())
    optimizer = torch.optim.Adam([
                    {'params': rec_params},
                    {'params': model.image.resnet.parameters(), 'lr': opts.lr*opts.freeImage }
                ], lr=opts.lr*opts.freeRecipe)

    if opts.restore:
        if os.path.isfile(opts.restore):
            print("=> loading checkpoint '{}'".format(opts.restore))
            checkpoint = torch.load(opts.restore)

            start_epoch = checkpoint['epoch']
            best_medr = checkpoint['best_medr']
            valtrack = checkpoint['valtrack']
            ckpts = checkpoint['ckpts']

            print('startepoch:', start_epoch)
            print('best medr:', best_medr)
            print('valtrack:', valtrack)
            print('ckpts:', ckpts)

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            opts.freeImage = checkpoint['freeImage']
            opts.freeRecipe = checkpoint['freeRecipe']
            change_lr(optimizer, opts.freeRecipe, opts.freeImage, opts.lr)
        else:
            print("=> no checkpoint found at '{}'".format(opts.restore))
            start_epoch = 0
            best_medr = float('inf')
            valtrack = 0
            ckpts = []
    else:
        start_epoch = 0
        best_medr = float('inf')
        valtrack = 0
        ckpts = []


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    # preparing the training loader
    train_dataset = MyDataSet(
            transforms.Compose([
                transforms.Resize(256), # rescale the image keeping the original aspect ratio
                transforms.CenterCrop(256), # we get only the center of that rescaled
                transforms.RandomCrop(224), # random crop within the center crop
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
            partition='train')

    train_batch_sampler = BatchSamplerTriplet(
                        indices_by_class = train_dataset.indices_by_class,
                        batch_size = opts.batch_size,
                        semantic_pc = opts.semantic_pc,
                        same_class_num = opts.same_class_num)

    train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_sampler=train_batch_sampler,
                    num_workers=opts.workers,
                    pin_memory=True)
    print('Training loader prepared.')

    # preparing validation loader
    valid_loader = torch.utils.data.DataLoader(
            MyDataSet(
                transforms.Compose([
                    transforms.Resize(256), # rescale the image keeping the original aspect ratio
                    transforms.CenterCrop(224), # we get only the center of that rescaled
                    transforms.ToTensor(),
                    normalize,
                ]),
                partition='valid'),
            batch_size= opts.batch_size,
            shuffle=False,
            num_workers=opts.workers,
            pin_memory=True)
    print( 'Validation loader prepared.')


    # preparing test loader
    test_loader = torch.utils.data.DataLoader(
            MyDataSet(
                transforms.Compose([
                    transforms.Resize(256), # rescale the image keeping the original aspect ratio
                    transforms.CenterCrop(224), # we get only the center of that rescaled
                    transforms.ToTensor(),
                    normalize,
                ]),
                partition='test'),
            batch_size= opts.batch_size,
            shuffle=False,
            num_workers=opts.workers,
            pin_memory=True)
    print( 'Test loader prepared.')

    data_time = 0.0
    batch_time = 0.0
    valid_time = 0.0
    test_time = 0.0

    start_time = time.time()
    end = time.time()

    for cur_epoch in range(start_epoch, opts.epochs):

        data_t, batch_t = train_epoch(model, criterion, train_loader, optimizer, cur_epoch)
        data_time += data_t
        batch_time += batch_t

        if (cur_epoch+1) % opts.valfreq == 0:
            end = time.time()

            val_medr = validate(model, criterion, valid_loader)
            is_best = False
            if val_medr <= best_medr:
                best_medr = val_medr
                is_best = True
                valtrack  = 0
                '''
                save_checkpoint({
                    'epoch': cur_epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_medr': best_medr,
                    'optimizer': optimizer.state_dict(),
                    'valtrack': valtrack,
                    'freeImage': opts.freeImage,
                    'freeRecipe': opts.freeRecipe}, ckpts)
                '''
            else:
                valtrack += 1
                if valtrack > opts.patience:
                    switch_phrase(optimizer, opts)
                    valtrack = 0
            valid_time += time.time() - end

            if is_best and cur_epoch > opts.startTest:
                end = time.time()
                test_lis_split = pkl.load(open(opts.test_lis_split, 'rb'))
                test(model, criterion, test_loader, test_lis_split)
                test_time += time.time() - end

        print('data_time:', data_time, 'batch_time:', batch_time, 'valid_time:', valid_time, 'test_time:', test_time)

    print('total time:', time.time()-start_time)

def train_epoch(model, criterion, train_loader, optimizer, epoch):
    print('epoch:', epoch)
    print('image ({imageLR}) - recipe ({recipeLR})'.format(
        imageLR=optimizer.param_groups[1]['lr'],
        recipeLR=optimizer.param_groups[0]['lr']))

    data_time = 0.0
    batch_time = 0.0
    end = time.time()

    model.train()

    for i, (inputs, classes) in enumerate(train_loader):
        torch.cuda.empty_cache()
        data_time += time.time() - end
        end = time.time()

        input_var = []
        for j in range(len(inputs)):
            input_var.append(torch.autograd.Variable(inputs[j]).cuda())

        class_var = list()
        for j in range(len(classes)):
            classes[j] = classes[j].cuda(non_blocking=True)
            class_var.append(torch.autograd.Variable(classes[j]))

        # compute output
        output = model(input_var[0], input_var[1], input_var[2], input_var[3]) #, input_var[4])

        loss = criterion(output[0], output[1], class_var[0], class_var[1])

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i % 100) == 0:
            print(loss.item(), 'epoch', epoch, 'batch', i, 'finish')

        batch_time += time.time() - end
        end = time.time()

    return data_time, batch_time

def validate(model, criterion, valid_loader):
    model.eval()

    with torch.no_grad():
        for i, (inputs, classes) in enumerate(valid_loader):
            input_var = []
            for j in range(len(inputs)):
                input_var.append(torch.autograd.Variable(inputs[j]).cuda(device=device))

            output = model(input_var[0],input_var[1], input_var[2], input_var[3]) #, input_var[4])

            if i==0:
                data0 = output[0].data.cpu().numpy()
                data1 = output[1].data.cpu().numpy()
            else:
                data0 = np.concatenate((data0,output[0].data.cpu().numpy()),axis=0)
                data1 = np.concatenate((data1,output[1].data.cpu().numpy()),axis=0)

    medR, recall = rank(opts, data0, data1)
    print('* Val medR {medR:.4f}\t'
        'Recall {recall}'.format(medR=medR, recall=recall))

    return medR

def test(model, criterion, test_loader, test_lis_split):
    print('testing...')

    model.eval()

    with torch.no_grad():
        for i, (inputs, classes) in enumerate(test_loader):
            input_var = []
            for j in range(len(inputs)):
                input_var.append(torch.autograd.Variable(inputs[j]).cuda())

            output = model(input_var[0],input_var[1], input_var[2], input_var[3]) # , input_var[4])

            if i==0:
                data0 = output[0].data.cpu().numpy()
                data1 = output[1].data.cpu().numpy()
            else:
                data0 = np.concatenate((data0,output[0].data.cpu().numpy()),axis=0)
                data1 = np.concatenate((data1,output[1].data.cpu().numpy()),axis=0)

    medR, recall = rank(opts, data0, data1, test_lis_split[0], 'image')
    print('* 1k im2recipe : Val medR {medR:.4f}\t'
        'Recall {recall}'.format(medR=medR, recall=recall))
    medR, recall = rank(opts, data0, data1, test_lis_split[0], 'recipe')
    print('* 1k recipe2im : Val medR {medR:.4f}\t'
        'Recall {recall}'.format(medR=medR, recall=recall))

    medR, recall = rank(opts, data0, data1, test_lis_split[1], 'image')
    print('* 5k im2recipe : Val medR {medR:.4f}\t'
        'Recall {recall}'.format(medR=medR, recall=recall))
    medR, recall = rank(opts, data0, data1, test_lis_split[1], 'recipe')
    print('* 5k recipe2im : Val medR {medR:.4f}\t'
        'Recall {recall}'.format(medR=medR, recall=recall))

    medR, recall = rank(opts, data0, data1, test_lis_split[2], 'image')
    print('* 10k im2recipe : Val medR {medR:.4f}\t'
        'Recall {recall}'.format(medR=medR, recall=recall))
    medR, recall = rank(opts, data0, data1, test_lis_split[2], 'recipe')
    print('* 10k recipe2im : Val medR {medR:.4f}\t'
        'Recall {recall}'.format(medR=medR, recall=recall))


def rank(opts, img_embeds, rec_embeds, rank_lis=None, embtype=None):
    if embtype == None:
        type_embedding = opts.embtype
    else:
        type_embedding = embtype
    num_examples = img_embeds.shape[0]

    # Ranker
    if rank_lis == None:
        valid_times = 15
        N = opts.medr
        ids = []
        for i in range(valid_times):
            ids.append(random.sample(range(0,num_examples), N))
    else:
        valid_times = len(rank_lis)
        ids = rank_lis

    # Modify to adapt to mini dataSet.
    for i in range(len(ids)):
        for j in range(len(ids[i])):
            ids[i][j] = ids[i][j] % num_examples

    glob_rank = []
    glob_recall = {1:0.0,5:0.0,10:0.0}
    for i in range(valid_times):
        
        img_sub = img_embeds[ids[i],:]
        rec_sub = rec_embeds[ids[i],:]

        num_id = len(ids[i])

        if type_embedding == 'image':
            sims = np.dot(img_sub,rec_sub.T) # for im2recipe
        else:
            sims = np.dot(rec_sub,img_sub.T) # for recipe2im

        med_rank = []
        recall = {1:0.0,5:0.0,10:0.0}

        for ii in range(num_id):
            sim = sims[ii,:]
            sorting = np.argsort(sim)[::-1].tolist()
            pos = sorting.index(ii)
            if (pos+1) == 1:
                recall[1]+=1
            if (pos+1) <=5:
                recall[5]+=1
            if (pos+1)<=10:
                recall[10]+=1
            med_rank.append(pos+1)

        for k in recall.keys():
            recall[k]=recall[k]/num_id

        med = np.median(med_rank)

        for k in recall.keys():
            glob_recall[k] += recall[k]
        glob_rank.append(med)

    for k in glob_recall.keys():
        glob_recall[k] = glob_recall[k]/valid_times

    return np.average(glob_rank), glob_recall

def save_checkpoint(state, ckpts):
    print('saving...')
    if not os.path.exists(opts.checkpoint):
        os.mkdir(opts.checkpoint)


    filename = os.path.join(opts.checkpoint,
        'model_e%03d_v-%.3f.pth.tar'%(state['epoch'],state['best_medr']) )

    ckpts.append(filename)

    while len(ckpts) > opts.maxCkpt:
        removeFile = ckpts.pop(0)
        if os.path.exists(removeFile):
            os.remove(removeFile)

    state['ckpts'] = ckpts
    torch.save(state, filename)


def switch_phrase(optimizer, opts):
    tmp_freeRecipe = opts.freeRecipe
    tmp_freeImage = opts.freeImage

    opts.freeRecipe = tmp_freeImage
    opts.freeImage = tmp_freeRecipe ^ tmp_freeImage

    change_lr(optimizer, opts.freeRecipe, opts.freeImage, opts.lr)

def change_lr(optimizer, freeRecipe, freeImage, lr):

    optimizer.param_groups[0]['lr'] = lr * freeRecipe
    optimizer.param_groups[1]['lr'] = lr * freeImage

    print('Initial recipe params lr: %f' % optimizer.param_groups[0]['lr'])
    print('Initial image params lr: %f' % optimizer.param_groups[1]['lr'])


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    setup_seed(2019)
    main()