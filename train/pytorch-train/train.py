# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import utils
import dataset
from keys import alphabet
#Alphabet = [e.encode('utf-8') for e in alphabet]
import models.crnn as crnn
import models.efficient_densecrnn as densecrnn
import distance
from tensorboard_logger import configure, log_value
configure("./logs/densecrnn", flush_secs=5)

parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', help='path to dataset',default='../data/newdata/train')
parser.add_argument('--valroot', help='path to dataset',default='../data/newdata/val')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=20, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=256, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.8, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--crnn', help="path to crnn (to continue training)",default='')#default='../pretrain-models/netCRNN.pth')
#parser.add_argument('--crnn', help="path to crnn (to continue training)",default='')
parser.add_argument('--alphabet', default=alphabet)
parser.add_argument('--experiment', help='Where to store samples and models',default='./save_model_four')
parser.add_argument('--displayInterval', type=int, default=50, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=1000, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=50, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=1000, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
print(opt)
ifUnicode=True
if opt.experiment is None:
    opt.experiment = 'expr'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.lmdbDataset(root=opt.trainroot)
assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
test_dataset = dataset.lmdbDataset(
    root=opt.valroot, transform=dataset.resizeNormalize((256, 32)))
test_dataset = dataset.lmdbDataset(
    root=opt.valroot)


ngpu = int(opt.ngpu)
nh = int(opt.nh)
alphabet = opt.alphabet
nclass = len(alphabet) + 1
nc = 1

converter = utils.strLabelConverter(alphabet)
criterion = CTCLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#crnn = crnn.CRNN(opt.imgH, nc, nclass, nh, ngpu)
crnn=densecrnn.DenseCrnnEfficient(nclass=nclass,nh=nh,growth_rate=12,block_config=(3,6,12,16),
                                  compression=0.5,
                                  num_init_features=24,bn_size=4,drop_rate=0,small=True)
crnn.apply(weights_init)
if opt.crnn != '':
    print('loading pretrained model from %s' % opt.crnn)
    crnn.load_state_dict(torch.load(opt.crnn))

print(crnn)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    crnn.cuda()
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)


def val(net, test_dataset, criterion, max_iter=2):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        test_dataset,  batch_size=opt.batchSize, num_workers=int(opt.workers),
        sampler=dataset.randomSequentialSampler(test_dataset, opt.batchSize),
        collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()
    test_distance=0
    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        if ifUnicode:
             cpu_texts = [ clean_txt(tx.decode('utf-8'))  for tx in cpu_texts]
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
       # preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred.strip() == target.strip():
                n_correct += 1
            # print(distance.levenshtein(pred.strip(), target.strip()))
            test_distance +=distance.nlevenshtein(pred.strip(), target.strip(),method=2)
    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):

        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
    accuracy = n_correct / float(max_iter * opt.batchSize)
    test_distance=test_distance/float(max_iter * opt.batchSize)
    testLoss = loss_avg.val()
    #print('Test loss: %f, accuray: %f' % (testLoss, accuracy))
    return testLoss,accuracy,test_distance

def clean_txt(txt):
    """
    filter char where not in alphabet with ' '
    """
    newTxt = u''
    for t in txt:
        if t in alphabet:
            newTxt+=t
        else:
            newTxt+=u' '
    return newTxt
    
def trainBatch(net, criterion, optimizer,flage=False):
    n_correct = 0
    train_distance=0

    data = train_iter.next()
    cpu_images, cpu_texts = data##decode utf-8 to unicode
    if ifUnicode:
        cpu_texts = [ clean_txt(tx.decode('utf-8'))  for tx in cpu_texts]
        
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()


    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
    for pred, target in zip(sim_preds, cpu_texts):
        if pred.strip() == target.strip():
            n_correct += 1
        train_distance +=distance.nlevenshtein(pred.strip(),target.strip(),method=2)
    train_accuracy = n_correct / float(batch_size)
    train_distance=train_distance/float(batch_size)


    if flage:
        lr = 0.0001
        optimizer = optim.Adadelta(crnn.parameters(), lr=lr)
    optimizer.step()
    return cost,train_accuracy,train_distance

num =0
lasttestLoss = 10000
testLoss = 10000
import os

def delete(path):
    """
    删除文件
    """
    import os
    import glob
    paths = glob.glob(path+'/*.pth')
    for p in paths:
        os.remove(p)
    
    
    
    
numLoss = 0##判断训练参数是否下降    
    
for epoch in range(opt.niter):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        #print('The step{} ........\n'.format(i))
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()
        #if numLoss>50:
        #    cost = trainBatch(crnn, criterion, optimizer,True)
        #    numLoss = 0
        #else:
        cost, train_accuracy, train_distance = trainBatch(crnn, criterion, optimizer)
        loss_avg.add(cost)
        i += 1

        #if i % opt.displayInterval == 0:
        #    print('[%d/%d][%d/%d] Loss: %f' %
        #          (epoch, opt.niter, i, len(train_loader), loss_avg.val()))
        #    loss_avg.reset()

        if i % opt.valInterval == 0:
            testLoss,accuracy,test_distance= val(crnn, test_dataset, criterion)
            localtime = time.asctime(time.localtime(time.time()))
            #print('Test loss: %f, accuray: %f' % (testLoss, accuracy))
            print("time:{},epoch:{},step:{},test loss:{},test acc:{},train loss:{},train acc:{},test dis:{},train dis:{}".format(localtime,epoch,num,testLoss,accuracy,loss_avg.val(),train_accuracy,test_distance,train_distance))
            log_value("test loss",float(testLoss),num)
            log_value("test accuracy",float(accuracy),num)
            log_value("train accuracy", float(train_accuracy), num)
            log_value("train loss",float(loss_avg.val()),num)
            log_value("test distanceloss", float(test_distance), num)
            log_value("train distanceloss",float(train_distance), num)
            loss_avg.reset()
        # do checkpointing
        num +=1
        #lasttestLoss = min(lasttestLoss,testLoss)
        
        if lasttestLoss >testLoss:
             print("The step {},last lost:{}, current: {},save model!".format(num,lasttestLoss,testLoss))
             lasttestLoss = testLoss
             #delete(opt.experiment)##删除历史模型
             torch.save(crnn.state_dict(), '{}/netCRNN{}.pth'.format(opt.experiment,str(accuracy)))
             numLoss = 0
        else:
            numLoss+=1
            
