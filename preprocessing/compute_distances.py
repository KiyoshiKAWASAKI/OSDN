# -*- coding: utf-8 -*-
###################################################################################################
# Copyright (c) 2016 , Regents of the University of Colorado on behalf of the University          #
# of Colorado Colorado Springs.  All rights reserved.                                             #
#                                                                                                 #
# Redistribution and use in source and binary forms, with or without modification,                #
# are permitted provided that the following conditions are met:                                   #
#                                                                                                 #
# 1. Redistributions of source code must retain the above copyright notice, this                  #
# list of conditions and the following disclaimer.                                                #
#                                                                                                 #
# 2. Redistributions in binary form must reproduce the above copyright notice, this list          #
# of conditions and the following disclaimer in the documentation and/or other materials          #
# provided with the distribution.                                                                 #
#                                                                                                 #
# 3. Neither the name of the copyright holder nor the names of its contributors may be            #
# used to endorse or promote products derived from this software without specific prior           #
# written permission.                                                                             #
#                                                                                                 #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY             #
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF         #
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          #
# THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,            #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF     #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,           #
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS           #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    #
#                                                                                                 #
# Author: Abhijit Bendale (abendale@vast.uccs.edu)                                                #
#                                                                                                 #
# If you use this code, please cite the following works                                           #
#                                                                                                 #
# A. Bendale, T. Boult “Towards Open Set Deep Networks” IEEE Conference on                        #
# Computer Vision and Pattern Recognition (CVPR), 2016                                            #
#                                                                                                 #
# Notice Related to using LibMR.                                                                  #
#                                                                                                 #
# If you use Meta-Recognition Library (LibMR), please note that there is a                        #
# difference license structure for it. The citation for using Meta-Recongition                    #
# library (LibMR) is as follows:                                                                  #
#                                                                                                 #
# Meta-Recognition: The Theory and Practice of Recognition Score Analysis                         #
# Walter J. Scheirer, Anderson Rocha, Ross J. Micheals, and Terrance E. Boult                     #
# IEEE T.PAMI, V. 33, Issue 8, August 2011, pages 1689 - 1695                                     #
#                                                                                                 #
# Meta recognition library is provided with this code for ease of use. However, the actual        #
# link to download latest version of LibMR code is: http://www.metarecognition.com/libmr-license/ #
###################################################################################################


import scipy as sp
import numpy as np
import sys
import os, glob
import scipy.spatial.distance as spd
from scipy.io import loadmat, savemat

nb_class = 293

model_dir = "/afs/crc.nd.edu/user/j/jhuang24/scratch_50/jhuang24/" \
            "models/msd_net/2022-02-13/known_only_cross_entropy/seed_0/"

train_feature_file_path = model_dir + "openmax_feature/train_features"
train_mean_file_save_path = model_dir + "openmax_feature/mean_files_train"
train_distance_save_path = model_dir + "openmax_feature/mean_distance_files_train"

valid_feature_file_path = model_dir + "openmax_feature/valid_features"
valid_mean_file_save_path = model_dir + "openmax_feature/mean_files_valid"
valid_distance_save_path = model_dir + "openmax_feature/mean_distance_files_valid"

test_known_feature_file_path = model_dir + "openmax_feature/test_known_features"
test_known_mean_file_save_path = model_dir + "openmax_feature/mean_files_test_known"
test_known_distance_save_path = model_dir + "openmax_feature/mean_distance_files_test_known"




def compute_channel_distances(mean_train_channel_vector, features):
    """
    Input:
    ---------
    mean_train_channel_vector : mean activation vector for a given class. 
                                It can be computed using MAV_Compute.py file
    features: features for the category under consideration
    category_name: synset_id

    Output:
    ---------
    channel_distances: dict of distance distribution from MAV for each channel. 
    distances considered are eucos, cosine and euclidean
    """

    eucos_dist, eu_dist, cos_dist = [], [], []

    # for channel in range(features[0].shape[0]):
    eu_channel, cos_channel, eu_cos_channel = [], [], []
    # compute channel specific distances
    for feat in features:
        eu_channel += [spd.euclidean(mean_train_channel_vector, feat)]
        cos_channel += [spd.cosine(mean_train_channel_vector, feat)]
        eu_cos_channel += [spd.euclidean(mean_train_channel_vector, feat)/200. +
                           spd.cosine(mean_train_channel_vector, feat)]

    eu_dist += [eu_channel]
    cos_dist += [cos_channel]
    eucos_dist += [eu_cos_channel]

    # convert all arrays as scipy arrays
    eucos_dist = sp.asarray(eucos_dist)
    eu_dist = sp.asarray(eu_dist)
    cos_dist = sp.asarray(cos_dist)

    # assertions for length check
    assert eucos_dist.shape[0] == 1
    assert eu_dist.shape[0] == 1
    assert cos_dist.shape[0] == 1
    assert eucos_dist.shape[1] == len(features)
    assert eu_dist.shape[1] == len(features)
    assert cos_dist.shape[1] == len(features)

    channel_distances = {'eucos': eucos_dist, 'cosine': cos_dist, 'euclidean':eu_dist}
    return channel_distances
    



def compute_distances(mav_fname,
                      category_name,
                      featurefilepath,
                      distance_save_path,
                      layer = 'fc8'):
    """
    Input:
    -------
    mav_fname : path to filename that contains mean activation vector
    labellist : list of labels from ilsvrc 2012
    category_name : synset_id

    """
    print("Processing class: ", category_name)

    featurefile_list = glob.glob('%s/%s/*.mat' % (featurefilepath, category_name))

    correct_features_top1 = []
    correct_features_top3 = []
    correct_features_top5 = []

    """
    Note: The correct feature could be empty
    """
    for featurefile in featurefile_list:
        img_arr = loadmat(featurefile)
        label = int(category_name)

        top_1 = img_arr['scores'].argmax()
        top_3 = np.argpartition(img_arr['scores'], -3)[-3:]
        top_5 = np.argpartition(img_arr['scores'], -5)[-5:]

        if top_1 == label:
            correct_features_top1 += [img_arr[layer]]

        if label in top_3:
            correct_features_top3 += [img_arr[layer]]

        if label in top_5:
            correct_features_top5 += [img_arr[layer]]

    if len(correct_features_top1) != 0:
        print("top1")
        mean_feature_vec_top_1 = loadmat(mav_fname + "/top_1/" + category_name + ".mat")[category_name]
        distance_top_1 = compute_channel_distances(mean_feature_vec_top_1,
                                                   correct_features_top1)
        savemat(distance_save_path + "/top_1/" + category_name + ".mat", distance_top_1)

    if len(correct_features_top3) != 0:
        print("top3")
        mean_feature_vec_top_3 = loadmat(mav_fname + "/top_3/" + category_name + ".mat")[category_name]
        distance_top_3 = compute_channel_distances(mean_feature_vec_top_3,
                                                    correct_features_top3)
        savemat(distance_save_path + "/top_3/" + category_name + ".mat", distance_top_3)

    if len(correct_features_top5) != 0:
        print("top5")
        mean_feature_vec_top_5 = loadmat(mav_fname + "/top_5/" + category_name + ".mat")[category_name]
        distance_top_5 = compute_channel_distances(mean_feature_vec_top_5,
                                                    correct_features_top5)
        savemat(distance_save_path + "/top_5/" + category_name + ".mat", distance_top_5)





if __name__ == "__main__":
    for i in range(nb_class):
        compute_distances(mav_fname=train_mean_file_save_path,
                          category_name=str(i).zfill(4),
                          featurefilepath=train_feature_file_path,
                          distance_save_path=train_distance_save_path)

    for i in range(nb_class):
        compute_distances(mav_fname=valid_mean_file_save_path,
                          category_name=str(i).zfill(4),
                          featurefilepath=valid_feature_file_path,
                          distance_save_path=valid_distance_save_path)

    for i in range(nb_class):
        compute_distances(mav_fname=test_known_mean_file_save_path,
                          category_name=str(i).zfill(4),
                          featurefilepath=test_known_feature_file_path,
                          distance_save_path=test_known_distance_save_path)
