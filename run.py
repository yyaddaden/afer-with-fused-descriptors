# -*- coding: UTF-8 -*-

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