# coding:utf-8

import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
#from genLineText import GenTextImage

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    #print (len(imagePathList) , len(labelList))
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    print '...................'
    env = lmdb.open(outputPath, map_size=1099511627776,map_async=True, metasync=False, writemap=True)
    
    cache = {}
    cnt = 1
    for i in xrange(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    env.sync()
    env.close()
    print('Created dataset with %d samples' % nSamples)


def read_text(path):
    
    with open(path) as f:
        text = f.read()
    text = text.strip()
    
    return text


import glob
if __name__ == '__main__':

    imagePathList=[]
    imgLabelList=[]
    ##lmdb 输出目录
    outputPath = '../data/newdata/train'
    path='/data/dataset/tpa_num_sogou/imgs/'
    txtpath = '/data/dataset/tpa_num_sogou/label.train.txt'
    with open(txtpath,'r') as f:
        for line in f:
            p=line.split(' ')
            if os.path.exists(path+p[0]):
                print path+p[0]
                imagePathList.append(path+p[0])
                p[1]=p[1].split('\n')[0]
                imgLabelList.append(p[1])
                print p[1]
            else:
                continue




    # imgLabelList = sorted(imgLabelLists,key = lambda x:len(x[1]))
    # imgPaths = [ p[0] for p in imgLabelList]
    # txtLists = [ p[1] for p in imgLabelList]
    
    createDataset(outputPath, imagePathList, imgLabelList, lexiconList=None, checkValid=True)




