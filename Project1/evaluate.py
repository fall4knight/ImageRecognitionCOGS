# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:43:50 2016

@author: Apurva Pathak
"""

import cv2
import numpy as np
import argparse

def evaluate(f_img,f_gt):
    accuracy=0.0
    img=cv2.imread(f_img,0);
    gt=cv2.imread(f_gt,0);

    (m,n)=np.shape(img)

    for i in range(m):
        for j in range(n):
            if(gt[i,j]>=100):
                gt[i,j]=1
            else:
                gt[i,j]=0
            if(img[i,j]>=100):
                img[i,j]=1
            else:
                img[i,j]=0
    f=3 #Filter size
    for i in range(m-f):
        for j in range(n-f):
            im=img[i:i+f,j:j+f]
            gtr=gt[i:i+f,j:j+f]
            if((np.sum(im)==0 and np.sum(gtr)!=0) or (np.sum(gtr)==0 and np.sum(im)!=0)):
                accuracy-=1
            else:
                accuracy+=1

    if accuracy<0:
        accuracy=0
    return accuracy/(m*n)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output_file_name',help='File name of the image containing the edges.')
    parser.add_argument('ground_truth_file_name',help='File name of the ground truth.')
    args=parser.parse_args()
    print('Accuracy: %f' %(evaluate(args.output_file_name,args.ground_truth_file_name)))


if __name__ == "__main__":
    main()
