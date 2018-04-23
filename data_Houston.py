# -- coding: utf-8 --

import scipy.io
import numpy as np
import random
from random import shuffle
import os
from skimage.util import pad
import scipy.ndimage
import math
import pandas as pd


patch_size = 25
num_band = 3

def prepare_data():
    Data = scipy.io.loadmat(os.path.join(os.getcwd(), 'data/Houston_pca.mat'))['Houston_pca']
    Label = scipy.io.loadmat(os.path.join(os.getcwd(), 'data/Houston_gt.mat'))['Houston_gt']

    # 获取数据size
    Height, Width, Band = Data.shape[0], Data.shape[1], Data.shape[2]
    # 获取分类数,np.unique()将返回一个不同元素从小到大排列的列表,类型0为未分类
    Num_Classes = len(np.unique(Label))-1

    # 归一化数据
    Data = Data.astype(float)
    for band in range(Band):
        Data[:, :, band] = (Data[:, :, band]-np.min(Data[:, :, band]))/(np.max(Data[:, :, band])-np.min(Data[:, :, band]))

    # 提前填充数据，在Height和Width上各添加patch_size(utils.py定义)-1个0像素
    Data_Padding = np.zeros((Height+int(patch_size-1), Width+int(patch_size-1), Band))
    for band in range(Band):
        # 中间填充数据
        Data_Padding[:, :, band] = pad(Data[:, :, band], int((patch_size-1)/2), 'symmetric')


    def Patch(height_index, width_index):
        """ function to extract patches from the orignal data """
        # 将数据切片，大小为patch_size*patch_size
        height_slice = slice(height_index, height_index + patch_size)
        width_slice = slice(width_index, width_index + patch_size)
        patch = Data_Padding[height_slice, width_slice, :]
        return np.array(patch)

    # 创建分类和分类索引号列表，列表内创建列表[[],[]...]
    Classes= []
    for k in range(Num_Classes):
        Classes.append([])

    All_Patches,  All_Labels =[], []
    for j in range(0, Width):
        for i in range(0, Height):
            curr_patch = Patch(i, j)
            curr_label = Label[i, j]
            if(curr_label!=0):
                Classes[curr_label - 1].append(curr_patch)
                All_Patches.append(curr_patch)
                All_Labels.append(curr_label-1)

    # 每个分类的数目
    Num_Each_Class=[]
    for k in range(Num_Classes):
        Num_Each_Class.append(len(Classes[k]))


    # 数据分离
    def DataDivide(Classes_k1, Num_Train_Each_Class_k):
        """ function to divide collected patches into training and test patches """
        # np.random.choice()从指定数组中生成指定个元素的数组，replace=True则生成的数可以重复
        idx = np.random.choice(len(Classes_k1), Num_Train_Each_Class_k, replace=False)
        train_patch = [Classes_k1[i] for i in idx]
        # 集合做差
        idx_test = np.setdiff1d(range(len(Classes_k1)),idx)
        test_patch = [Classes_k1[i] for i in idx_test]
        return train_patch, test_patch

    # 制作训练集和测试集
    Num_Train_Each_Class = [50]*Num_Classes

    Num_Test_Each_Class = list(np.array(Num_Each_Class) - np.array(Num_Train_Each_Class))
    Train_Patch, Train_Label, Test_Patch, Test_Label = [], [], [], []

    for k in range(Num_Classes):
        # 生成训练集，数据集
        train_patch, test_patch = DataDivide(Classes[k], Num_Train_Each_Class[k])
        #Make training and test splits
        Train_Patch.append(train_patch)    # patches_of_current_class[:-test_split_size]
        Test_Patch.extend(test_patch)    # patches_of_current_class[-test_split_size:]
        # np.full()创建常数组
        Test_Label.extend(np.full(Num_Test_Each_Class[k], k, dtype=int))

    Train_Label = []
    for k in range(Num_Classes):
        Train_Label.append([k]*Num_Train_Each_Class[k])


    OS_Aug_Num_Training_Each = []
    for k in range(Num_Classes):
        OS_Aug_Num_Training_Each.append(len(Train_Label[k]))

    Temp1, Temp2 = [], []
    for k in range(Num_Classes):
        Temp1.extend(Train_Patch[k])
        Temp2.extend(Train_Label[k])
    Train_Patch = Temp1
    Train_Label = Temp2

    def convertToOneHot(vector, num_classes=None):
        """
        Converts an input 1-D vector of integers into an output
        2-D array of one-hot vectors, where an i'th input value
        of j will set a '1' in the i'th row, j'th column of the
        output array.

        Example:
            v = np.array((1, 0, 4))
            one_hot_v = convertToOneHot(v)
            print one_hot_v

            [[0 1 0 0 0]
             [1 0 0 0 0]
             [0 0 0 0 1]]
        """

        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0

        if num_classes is None:
            num_classes = np.max(vector)+1
        else:
            assert num_classes > 0
            assert num_classes >= np.max(vector)

        result = np.zeros(shape=(len(vector), num_classes))
        result[np.arange(len(vector)), vector] = 1
        return result.astype(int)

    # Convert the labels to One-Hot vector
    # onehot编码
    Train_Patch = np.array(Train_Patch)
    Test_Patch = np.array(Test_Patch)
    Train_Label = np.array(Train_Label)
    Test_Label = np.array(Test_Label)
    All_Patches = np.array(All_Patches)
    All_Labels = np.array(All_Labels)

    test_ind = {}
    test_ind['TestLabel'] = Test_Label
    scipy.io.savemat(os.path.join(os.getcwd(), 'result/TestLabel.mat'), test_ind)

    Train_Label = convertToOneHot(Train_Label,num_classes=Num_Classes)
    Test_Label = convertToOneHot(Test_Label,num_classes=Num_Classes)
    All_Labels = convertToOneHot(All_Labels, num_classes=Num_Classes)

    # Data Summary
    df = pd.DataFrame(np.random.randn(Num_Classes, 4),
                      columns=['Total', 'Training', 'OS&Aug', 'Testing'])
    df['Total'] = Num_Each_Class
    df['Training'] = Num_Train_Each_Class
    df['OS&Aug'] = OS_Aug_Num_Training_Each
    df['Testing'] = Num_Test_Each_Class
    num_train = len(Train_Patch)
    num_test = len(Test_Patch)
    print("=======================================================================")
    print("Data Summary")
    print("=======================================================================")
    print('The size of the original HSI data is (%d,%d,%d)' % (Height, Width, Band))
    print('The size of Training data is (%d)' % (num_train))
    print('The size of Test data is (%d)' % (num_test))
    print('The size of each sample is (%d,%d,%d)' % (patch_size, patch_size, Band))
    print('-----------------------------------------------------------------------')
    print("The Data Division is")
    print(df)
    return Train_Patch, Test_Patch, Train_Label, Test_Label, All_Patches, All_Labels





