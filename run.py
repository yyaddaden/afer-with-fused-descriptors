# -*- coding: UTF-8 -*-

"""
Author: Prof. Yacine Yaddaden, Ph. D. (Université du Québec à Rimouski)

IMPORTANT : 
If used in the context of research project, please cite the following paper :

@article{YADDADEN2023200166,
    title = {An efficient facial expression recognition system with appearance-based fused descriptors},
    journal = {Intelligent Systems with Applications},
    volume = {17},
    pages = {200166},
    year = {2023},
    issn = {2667-3053},
    doi = {https://doi.org/10.1016/j.iswa.2022.200166},
    url = {https://www.sciencedirect.com/science/article/pii/S266730532200103X},
    author = {Yacine Yaddaden}
}
"""

import os, sys

from ISWA2022 import ISWA2022

if __name__ == "__main__":
    afer = ISWA2022(emotions=7)

    if(sys.argv[1] == "-e"):
        # EVALUATING
        in_db_path = sys.argv[2] # "DB - JAFFE"
        afer.evaluate(in_db_path, nbr_cpnt_lbp=70, nbr_cpnt_hog=65)

    elif(sys.argv[1] == "-t"):
        # TRAINING
        in_db_path = sys.argv[2] # "DB - JAFFE"
        emotions = ["FE", "SU", "HA", "DI", "AN", "SA"]
        feature_vectors_lbp = []
        feature_vectors_hog = []
        labels = []

        for emotion in emotions:
            images = os.listdir(f"{in_db_path}/{emotion}/")

            for image in images:
                labels.append(emotion)
                in_img = afer.preprocess(f"{in_db_path}/{emotion}/{image}")
                local_lbp_feat_vect, global_hog_feat_vect = afer.describe(in_img)
                feature_vectors_lbp.append(local_lbp_feat_vect)
                feature_vectors_hog.append(global_hog_feat_vect)

        afer.train(feature_vectors_lbp, feature_vectors_hog, labels, nbr_cpnt_lbp=70, nbr_cpnt_hog=65)

    elif(sys.argv[1] == "-r"):
        # RECOGNIZING
        in_img_filename = sys.argv[2] # "images/HA.tiff"
        in_img = afer.preprocess(in_img_filename)
        local_lbp_feat_vect, global_hog_feat_vect = afer.describe(in_img)

        emotion, confidence = afer.recognize(local_lbp_feat_vect, global_hog_feat_vect)
        print(f"The recognized emotion from ({in_img_filename}) is : {emotion} ({confidence}%)")

    else:
        print("Error ! wrong or no argument provided !")