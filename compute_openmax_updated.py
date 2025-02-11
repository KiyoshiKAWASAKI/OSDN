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

# Modified by Jin Huang
# Modify to process features and obtain openMax score

import os, sys, pickle, glob
import os.path as path
import argparse
import scipy.spatial.distance as spd
import scipy as sp
import numpy as np
from scipy.io import loadmat
import pickle

from openmax_utils import *
from evt_fitting import weibull_tailfitting, query_weibull
from libMR import libmr


# ---------------------------------------------------------------------------------
# params and configurations

# model_seed = 0
# NB_FINAL_CLASSES = 277

# model_seed = 1
# NB_FINAL_CLASSES = 279

# model_seed = 2
# NB_FINAL_CLASSES = 276

# model_seed = 3
# NB_FINAL_CLASSES = 277

model_seed = 4
NB_FINAL_CLASSES = 276

WEIBULL_TAIL_SIZE = 1000

NCLASSES = 293
NCHANNELS = 1
ALPHA_RANK = 10


# TODO: These are for one img only now
base_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_50/jhuang24/models/msd_net/" \
            "2022-02-13/known_only_cross_entropy/seed_" + str(model_seed) + "/openmax_feature"
distance_path = os.path.join(base_path, "mean_distance_files_train/top_1")
mean_path = os.path.join(base_path, "mean_files_train/top_1")

save_label_mapping_path = os.path.join(base_path, "label_mapping.pkl")

train_img_base_dir = os.path.join(base_path, "train_features")
valid_img_base_dir = os.path.join(base_path, "valid_features")
test_known_img_base_dir = os.path.join(base_path, "test_known_features")
test_unknown_img_base_dir = os.path.join(base_path, "test_unknown_features")

train_result_save_path = os.path.join(base_path, "train_results_tail_size_" + str(WEIBULL_TAIL_SIZE) + ".npy")
valid_result_save_path = os.path.join(base_path, "valid_results_tail_size_" + str(WEIBULL_TAIL_SIZE) + ".npy")
test_known_result_save_path = os.path.join(base_path, "test_known_results_tail_size_" + str(WEIBULL_TAIL_SIZE) + ".npy")
test_unknown_result_save_path = os.path.join(base_path, "test_unknown_results_tail_size_" + str(WEIBULL_TAIL_SIZE) + ".npy")

# ---------------------------------------------------------------------------------
def computeOpenMaxProbability(openmax_fc8, openmax_score_u):
    """ Convert the scores in probability value using openmax

    Input:
    ---------------
    openmax_fc8 : modified FC8 layer from Weibull based computation
    openmax_score_u : degree

    Output:
    ---------------
    modified_scores : probability values modified using OpenMax framework,
    by incorporating degree of uncertainity/openness for a given class

    """
    prob_scores, prob_unknowns = [], []

    for channel in range(NCHANNELS):
        channel_scores, channel_unknowns = [], []
        for category in range(NB_FINAL_CLASSES):
            channel_scores += [sp.exp(openmax_fc8[channel, category])]

        total_denominator = sp.sum(sp.exp(openmax_fc8[channel, :])) + sp.exp(sp.sum(openmax_score_u[channel, :]))
        prob_scores += [channel_scores / total_denominator]
        prob_unknowns += [sp.exp(sp.sum(openmax_score_u[channel, :])) / total_denominator]

    prob_scores = sp.asarray(prob_scores)
    prob_unknowns = sp.asarray(prob_unknowns)

    scores = sp.mean(prob_scores, axis=0)
    unknowns = sp.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]

    print(len(modified_scores))
    assert len(modified_scores) == NB_FINAL_CLASSES + 1
    return modified_scores


# ---------------------------------------------------------------------------------
def recalibrate_scores(weibull_model,
                       imgarr,
                       label_dict,
                       layer='fc8',
                       alpharank=10,
                       distance_type='eucos'):
    """
    Given FC8 features for an image, list of weibull models for each class,
    re-calibrate scores

    Input:
    ---------------
    weibull_model : pre-computed weibull_model obtained from weibull_tailfitting() function
    labellist : ImageNet 2012 labellist
    imgarr : features for a particular image extracted using caffe architecture

    Output:
    ---------------
    openmax_probab: Probability values for a given class computed using OpenMax
    softmax_probab: Probability values for a given class computed using SoftMax (these
    were precomputed from caffe architecture. Function returns them for the sake
    of convienence)

    """

    imglayer = imgarr[layer]
    ranked_list = imgarr['scores'].argsort().ravel()[::-1]
    alpha_weights = [((alpharank + 1) - i) / float(alpharank) for i in range(1, alpharank + 1)]
    ranked_alpha = sp.zeros(NCLASSES)
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    # Now recalibrate each fc8 score for each channel and for each class
    # to include probability of unknown
    openmax_fc8, openmax_score_u = [], []

    for channel in range(NCHANNELS):
        channel_scores = imglayer[channel, :]
        openmax_fc8_channel = []
        openmax_fc8_unknown = []

        for category in range(NCLASSES):
            try:
                categoryid = label_dict[str(category).zfill(4)]
                # get distance between current channel and mean vector
                category_weibull = query_weibull(original_category=category,
                                                    category_name=categoryid,
                                                    weibull_model=weibull_model,
                                                    distance_type=distance_type)

                channel_distance = compute_distance(channel_scores, channel, category_weibull[0],
                                                    distance_type=distance_type)


                # obtain w_score for the distance and compute probability of the distance
                # being unknown wrt to mean training vector and channel distances for
                # category and channel under consideration
                wscore = category_weibull[2][channel].w_score(channel_distance)
                modified_fc8_score = channel_scores[categoryid] * (1 - wscore * ranked_alpha[categoryid])
                openmax_fc8_channel += [modified_fc8_score]
                openmax_fc8_unknown += [channel_scores[categoryid] - modified_fc8_score]
            except:
                pass

        # gather modified scores fc8 scores for each channel for the given image
        openmax_fc8 += [openmax_fc8_channel]
        openmax_score_u += [openmax_fc8_unknown]

    openmax_fc8 = sp.asarray(openmax_fc8)
    openmax_score_u = sp.asarray(openmax_score_u)

    # Pass the recalibrated fc8 scores for the image into openmax
    openmax_probab = computeOpenMaxProbability(openmax_fc8, openmax_score_u)
    softmax_probab = imgarr['scores'].ravel()
    return sp.asarray(openmax_probab), sp.asarray(softmax_probab)




if __name__ == "__main__":
    # Fit weibull with 277 classes (those who have correct training predictions)
    weibull_model, label_mapping, missing_classes = weibull_tailfitting(mean_path,
                                                                        distance_path,
                                                                        list(range(NCLASSES)),
                                                                        tailsize=WEIBULL_TAIL_SIZE)
    print ("Completed Weibull fitting on %s models" % len(weibull_model.keys()))
    print ("Missing %s classes in Weibull" % len(missing_classes))

    print(label_mapping)

    with open(save_label_mapping_path, 'wb') as f:
        pickle.dump(label_mapping, f)

    # # Compute both openmax and softmax
    # # TODO: Training
    # train_data_result = []
    #
    # for i in range(NCLASSES):
    #     if os.path.exists(os.path.join(train_img_base_dir, str(i).zfill(4))):
    #         all_imgs = os.listdir(os.path.join(train_img_base_dir, str(i).zfill(4)))
    #
    #         for one_img in all_imgs:
    #             one_img_path = os.path.join(os.path.join(train_img_base_dir, str(i).zfill(4)), one_img)
    #             imgarr = loadmat(one_img_path)
    #
    #             openmax, softmax = recalibrate_scores(weibull_model=weibull_model,
    #                                                   label_dict=label_mapping,
    #                                                   imgarr=imgarr)
    #
    #             train_data_result.append([i, np.argmax(softmax), np.argmax(openmax)])
    #
    # np.save(train_result_save_path, train_data_result)
    #
    #
    # # TODO: Validation
    # valid_data_result = []
    #
    # for i in range(NCLASSES):
    #
    #     if os.path.exists(os.path.join(valid_img_base_dir, str(i).zfill(4))):
    #         all_imgs = os.listdir(os.path.join(valid_img_base_dir, str(i).zfill(4)))
    #
    #         for one_img in all_imgs:
    #             one_img_path = os.path.join(os.path.join(valid_img_base_dir, str(i).zfill(4)), one_img)
    #             imgarr = loadmat(one_img_path)
    #
    #             openmax, softmax = recalibrate_scores(weibull_model=weibull_model,
    #                                                   label_dict=label_mapping,
    #                                                   imgarr=imgarr)
    #
    #             valid_data_result.append([i, np.argmax(softmax), np.argmax(openmax)])
    #
    # np.save(valid_result_save_path, valid_data_result)
    #
    #
    # # TODO: Test known
    # test_known_data_result = []
    #
    # for i in range(NCLASSES):
    #     if os.path.exists(os.path.join(test_known_img_base_dir, str(i).zfill(4))):
    #         all_imgs = os.listdir(os.path.join(test_known_img_base_dir, str(i).zfill(4)))
    #
    #         for one_img in all_imgs:
    #             one_img_path = os.path.join(os.path.join(test_known_img_base_dir, str(i).zfill(4)), one_img)
    #             imgarr = loadmat(one_img_path)
    #
    #             openmax, softmax = recalibrate_scores(weibull_model=weibull_model,
    #                                                   label_dict=label_mapping,
    #                                                   imgarr=imgarr)
    #
    #             test_known_data_result.append([i, np.argmax(softmax), np.argmax(openmax)])
    #
    # np.save(test_known_result_save_path, test_known_data_result)
    #
    #
    # # TODO: Test unknown
    # test_unknown_data_result = []
    #
    # all_imgs = os.listdir(os.path.join(test_unknown_img_base_dir, str(NCLASSES+1).zfill(4)))
    #
    # for one_img in all_imgs:
    #     one_img_path = os.path.join(os.path.join(test_unknown_img_base_dir, str(NCLASSES+1).zfill(4)), one_img)
    #     imgarr = loadmat(one_img_path)
    #
    #     openmax, softmax = recalibrate_scores(weibull_model=weibull_model,
    #                                           label_dict=label_mapping,
    #                                           imgarr=imgarr)
    #
    #     test_unknown_data_result.append([NCLASSES+1, np.argmax(softmax), np.argmax(openmax)])
    #
    # np.save(test_unknown_result_save_path, test_unknown_data_result)

