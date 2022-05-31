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
#                                                                                                 #
# Author: Abhijit Bendale (abendale@vast.uccs.edu)                                                #
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

import sys
import glob
import time
import scipy as sp
import numpy as np
from scipy.io import loadmat, savemat


nb_class = 293

model_dir = "/afs/crc.nd.edu/user/j/jhuang24/scratch_50/jhuang24/models/msd_net/2022-02-13/" \
             "known_only_cross_entropy/seed_4/"

train_feature_file_path = model_dir + "openmax_feature/train_features"
train_mean_file_save_path = model_dir + "openmax_feature/mean_files_train"

valid_feature_file_path = model_dir + "openmax_feature/valid_features"
valid_mean_file_save_path = model_dir + "openmax_feature/mean_files_valid"

test_known_feature_file_path = model_dir + "openmax_feature/test_known_features"
test_known_mean_file_save_path = model_dir + "openmax_feature/mean_files_test_known"


def getlabellist(fname):
    imagenetlabels = open(fname, 'r').readlines()
    labellist  = [i.split(' ')[0] for i in imagenetlabels]        
    return labellist




def compute_channel_mav(correct_features,
                        class_index,
                        category,
                        mean_file_save_path):
    """

    :param correct_features:
    :param category_name:
    :param mean_file_save_path:
    :return:
    """
    channel_mean_vec = []

    for channelid in range(correct_features[0].shape[0]):
        channel = []
        for feature in correct_features:
            channel += [feature[channelid, :]]
        channel = sp.asarray(channel)
        assert len(correct_features) == channel.shape[0]
        # Gather mean over each channel, to get mean channel vector
        channel_mean_vec += [sp.mean(channel, axis=0)]

    # this vector contains mean computed over correct classifications
    # for each channel separately
    channel_mean_vec = sp.asarray(channel_mean_vec)

    save_path_full = mean_file_save_path + "/" + category + '/%s.mat' % (class_index)
    print("Saving mean activate vector to: ", save_path_full)
    savemat(save_path_full, {class_index: channel_mean_vec})




def compute_mean_vector(category_name,
                        feature_path,
                        save_path,
                        layer = 'fc8'):
    print("Processing: ", category_name)

    featurefile_list = glob.glob('%s/%s/*.mat' % (feature_path, category_name))
    
    # gather all the training samples for which predicted category
    # was the category under consideration
    correct_features_top1 = []
    correct_features_top3 = []
    correct_features_top5 = []

    for featurefile in featurefile_list:
        try:
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

        except TypeError:
            continue

    # Now compute channel wise mean vector
    try:
        compute_channel_mav(correct_features=correct_features_top1,
                            class_index=category_name,
                            category="top_1",
                            mean_file_save_path=save_path)
    except:
        print("no correct top 1 predictions")
        pass

    try:
        compute_channel_mav(correct_features=correct_features_top3,
                            class_index=category_name,
                            category="top_3",
                            mean_file_save_path=save_path)
    except:
        print("no correct top 3 predictions")
        pass

    try:
        compute_channel_mav(correct_features=correct_features_top5,
                            class_index=category_name,
                            category="top_5",
                            mean_file_save_path=save_path)
    except:
        print("no correct top5 predictions")
        pass


if __name__ == "__main__":
    print("*" * 50)
    for i in range(nb_class):
        st = time.time()
        compute_mean_vector(category_name=str(i).zfill(4),
                            feature_path=train_feature_file_path,
                            save_path=train_mean_file_save_path)
        print ("Total time %s secs" % (time.time() - st))

    print("*" * 50)
    for i in range(nb_class):
        st = time.time()
        compute_mean_vector(category_name=str(i).zfill(4),
                            feature_path=valid_feature_file_path,
                            save_path=valid_mean_file_save_path)
        print ("Total time %s secs" % (time.time() - st))

    print("*" * 50)
    for i in range(nb_class):
        st = time.time()
        compute_mean_vector(category_name=str(i).zfill(4),
                            feature_path=test_known_feature_file_path,
                            save_path=test_known_mean_file_save_path)
        print ("Total time %s secs" % (time.time() - st))
