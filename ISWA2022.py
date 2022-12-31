# -*- coding: UTF-8 -*-

from os import listdir, path, mkdir
from math import sqrt, acos, degrees
from joblib import dump, load

from dlib import get_frontal_face_detector, shape_predictor
import numpy as np
from skimage import io, color, transform, img_as_ubyte, exposure, feature
from sklearn import (
    svm,
    decomposition,
    preprocessing,
    model_selection,
    metrics,
    calibration,
)


class ISWA2022:
    def __init__(self, emotions=7):
        print("-- Start --")

        if emotions == 7:
            self.emotions = ["FE", "SU", "HA", "DI", "AN", "SA", "NE"]
        else:
            self.emotions = ["FE", "SU", "HA", "DI", "AN", "SA"]

        self.hog_parameters = {
            "orientations": 8,
            "pixels_per_cell": (4, 4),
            "cells_per_block": (1, 1),
        }

        self.lbp_parameters = {
            "nbr_points": 48,
            "radius": 6,
            "method": "nri_uniform",
        }

        self.detector = get_frontal_face_detector()
        self.predictor = shape_predictor("shape_predictor_68_face_landmarks.dat")

    # PRIVATE METHODS

    """ extract the 68 face landmarks
        - shape_predictor_68_face_landmarks.dat file is required

        INPUT <- input image
        OUTPUT -> dictionary containing X and Y coordinates 
    """

    def __landmarks_extract(self, img_in):
        detect_faces = self.detector(img_in, 1)
        shape = self.predictor(img_in, detect_faces[0])

        landmarks = {"X": [], "Y": []}

        for idx in range(68):
            landmarks["X"].append(shape.part(idx).x)
            landmarks["Y"].append(shape.part(idx).y)

        return landmarks

    """ perfrom face alignment using eyes positions

        INPUT <- input image
        OUTPUT -> aligned image 
    """

    def __face_align(self, img_in):
        landmarks = self.__landmarks_extract(img_in)

        x1 = ((landmarks["X"][39] - landmarks["X"][36]) / 2) + landmarks["X"][36]
        y1 = ((landmarks["Y"][40] - landmarks["Y"][37]) / 2) + landmarks["Y"][37]

        x2 = ((landmarks["X"][45] - landmarks["X"][42]) / 2) + landmarks["X"][42]
        y2 = ((landmarks["Y"][47] - landmarks["Y"][44]) / 2) + landmarks["Y"][44]

        line_straight = sqrt(pow((x1 - x2), 2))
        line_curved = sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))
        angle = acos(float(line_straight) / float(line_curved))
        angle = degrees(-angle if (y1 > y2) else angle)

        img_out = transform.rotate(img_in, angle)
        img_out = img_as_ubyte(img_out)

        return img_out

    """ extract the face region (ROI)

        INPUT <- input image | face landmarks 
        OUTPUT -> face image (128 x 128 pixels)
    """

    def __face_extract(self, img_in, landmarks):
        X_min = min(landmarks["X"])
        Y_min = min(landmarks["Y"])
        X_max = max(landmarks["X"])
        Y_max = max(landmarks["Y"])

        img_out = img_in[Y_min:Y_max, X_min:X_max]
        img_out = transform.resize(img_out, (128, 128))

        img_out = exposure.equalize_hist(img_out)

        return img_out

    """ perform recognition of several feature vectors

        INPUT <- input feature vectors | corresponding labels | scaler | pca | clf(SVM)
        OUTPUT -> accuracy | confusion matrix
    """

    def __test(
        self,
        in_feature_vectors_lbp,
        in_feature_vectors_hog,
        in_labels,
        scaler_lbp,
        pca_lbp,
        scaler_hog,
        pca_hog,
        clf,
    ):
        in_feature_vectors_lbp = scaler_lbp.transform(in_feature_vectors_lbp)
        in_feature_vectors_lbp = pca_lbp.transform(in_feature_vectors_lbp)

        in_feature_vectors_hog = scaler_hog.transform(in_feature_vectors_hog)
        in_feature_vectors_hog = pca_hog.transform(in_feature_vectors_hog)

        in_feature_vectors = []
        for idx in range(len(in_feature_vectors_lbp)):
            in_feature_vectors.append(
                in_feature_vectors_lbp[idx].tolist()
                + in_feature_vectors_hog[idx].tolist()
            )

        out_predicted = clf.predict(in_feature_vectors)

        conf_mat = metrics.confusion_matrix(
            in_labels, out_predicted, labels=self.emotions
        )
        accuracy = metrics.accuracy_score(in_labels, out_predicted)

        return accuracy, conf_mat

    """ compute the average confusion matrix

        INPUT <- all the confusion matrices (10 folds cross-validaton)
        OUTPUT -> the final confusion matrix
    """

    def __mean_conf_mat(self, conf_mats):
        conf_mat = np.zeros(conf_mats[0].shape)

        for element in conf_mats:
            for i in range(conf_mats[0].shape[0]):
                for j in range(conf_mats[0].shape[0]):
                    conf_mat[i][j] += element[i][j]

        conf_mat = conf_mat.astype("float") / conf_mat.sum(axis=1)[:, np.newaxis]

        for idx in range(len(self.emotions)):
            for jdx in range(len(self.emotions)):
                conf_mat[idx][jdx] = "%0.2f" % (conf_mat[idx][jdx] * 100)

        return conf_mat.tolist()

    # PUBLIC METHODS

    """ (1) apply several preprocessing tasks
        - landmark extraction
        - extract the face region
        - apply histogram equalization

        INPUT <- input image filename
        OUTPUT -> preprocessed image
    """

    def preprocess(self, in_image_filename):
        # print("-- PRE-PROCESSING --")
        img_in = io.imread(in_image_filename, plugin="pil")

        if len(img_in.shape) == 3:
            img_in = color.rgb2gray(img_in)
            img_in = img_as_ubyte(img_in)

        img_out = self.__face_align(img_in)
        landmarks = self.__landmarks_extract(img_out)
        img_out = self.__face_extract(img_out, landmarks)

        return img_out

    """ (2) extract the feature vector
        - using g-HOG and l-LBP
        - combine both feature vectors

        INPUT <- input image
        OUTPUT -> extracted feature vectors (LBP and HOG)
    """

    def describe(self, img_in):
        # print("-- FEATURE EXATRACTION --")
        # l-LBP
        local_lbp_feat_vect = []

        face_regions = [
            img_in[0:64, 0:64],
            img_in[0:64, 64:128],
            img_in[64:128, 32:96],
            img_in[32:80, 32:96],
        ]

        for face_region in face_regions:
            lbp_image = feature.local_binary_pattern(
                face_region,
                P=self.lbp_parameters["nbr_points"],
                R=self.lbp_parameters["radius"],
                method=self.lbp_parameters["method"],
            )
            nbr_bins = int(lbp_image.max() + 1)
            lbp_hist, _ = np.histogram(lbp_image, bins=nbr_bins, range=(0, nbr_bins))
            local_lbp_feat_vect += lbp_hist.tolist()

        # g-HOG
        global_hog_feat_vect = feature.hog(
            img_in,
            orientations=self.hog_parameters["orientations"],
            pixels_per_cell=self.hog_parameters["pixels_per_cell"],
            cells_per_block=self.hog_parameters["cells_per_block"],
            visualize=False,
            feature_vector=True,
        )

        return local_lbp_feat_vect, global_hog_feat_vect.tolist()

    """ (3) reduce the size of the database & train the model 
        - apply the PCA
        - train the SVM classifier
        - using the data (self.data) and labels (self.labels)
        - save the generated models (SVM & PCA)

        INPUT <- input train data LBP & HOG | input train labels | number of components for LBP & HOG
        OUTPUT -> save the models SVM, PCA & scaler for HOG and LBP
    """

    def train(
        self,
        in_train_data_lbp,
        in_train_data_hog,
        in_train_labels,
        nbr_cpnt_lbp,
        nbr_cpnt_hog,
        save=True,
    ):
        # print("-- TRAINING --")
        self.nbr_cpnt_lbp = nbr_cpnt_lbp
        self.nbr_cpnt_hog = nbr_cpnt_hog

        clf = svm.SVC(
            C=1.0,
            kernel="linear",
            random_state=10,
            decision_function_shape="ovr",
            probability=True,
        )

        scaler_lbp = preprocessing.MinMaxScaler(feature_range=(0, 1))
        in_train_data_lbp = scaler_lbp.fit_transform(in_train_data_lbp)
        pca_lbp = decomposition.PCA(n_components=nbr_cpnt_lbp)
        in_train_data_lbp = pca_lbp.fit_transform(in_train_data_lbp)
        in_train_data_lbp = in_train_data_lbp.tolist()

        scaler_hog = preprocessing.MinMaxScaler(feature_range=(0, 1))
        in_train_data_hog = scaler_hog.fit_transform(in_train_data_hog)
        pca_hog = decomposition.PCA(n_components=nbr_cpnt_hog)
        in_train_data_hog = pca_hog.fit_transform(in_train_data_hog)
        in_train_data_hog = in_train_data_hog.tolist()

        in_train_data = []
        for idx in range(len(in_train_data_lbp)):
            in_train_data.append(in_train_data_lbp[idx] + in_train_data_hog[idx])

        clf = calibration.CalibratedClassifierCV(clf)
        clf.fit(in_train_data, in_train_labels)

        if save:
            if not path.exists("models"):
                mkdir("models")

            dump(scaler_lbp, "models/scaler_lbp.joblib")
            dump(scaler_hog, "models/scaler_hog.joblib")
            dump(pca_lbp, "models/pca_lbp.joblib")
            dump(pca_hog, "models/pca_hog.joblib")
            dump(clf, "models/svm.joblib")
        else:
            return scaler_lbp, scaler_hog, pca_lbp, pca_hog, clf

    """ recognize the emotion from the input image
        - apply the PCA (existing model)
        - apply the SVM (existing model)

        INPUT <- input feature vectors LBP & HOG
        OUTPUT -> the recognized emotion "FE", "SU", "HA", "DI", "AN", "SA" or "NE"
    """

    def recognize(self, in_feature_vector_lbp, in_feature_vector_hog):
        # print("-- RECOGNIZING --")
        in_feature_vector_lbp = np.array(in_feature_vector_lbp).reshape(1, -1)
        scaler_lbp = load("models/scaler_lbp.joblib")
        in_feature_vector_lbp = scaler_lbp.transform(in_feature_vector_lbp)
        pca_lbp = load("models/pca_lbp.joblib")
        in_feature_vector_lbp = pca_lbp.transform(in_feature_vector_lbp)[0].tolist()

        in_feature_vector_hog = np.array(in_feature_vector_hog).reshape(1, -1)
        scaler_hog = load("models/scaler_hog.joblib")
        in_feature_vector_hog = scaler_hog.transform(in_feature_vector_hog)
        pca_hog = load("models/pca_hog.joblib")
        in_feature_vector_hog = pca_hog.transform(in_feature_vector_hog)[0].tolist()

        in_feature_vector = in_feature_vector_lbp + in_feature_vector_hog

        in_feature_vector = np.array(in_feature_vector).reshape(1, -1)

        clf = load("models/svm.joblib")

        confidence = "%0.2f" % (max(clf.predict_proba(in_feature_vector)[0]) * 100)
        predicted_emotion = clf.predict(in_feature_vector)[0]

        return predicted_emotion, confidence

    """ evaluate the perfromance of the afer system on a benchmark database
        - perform preprocessing
        - perform feature extraction (l-LBP & g-HOG)
        - perform dimensionality reduction (PCA)
        - perform classification (multi-class SVM)

        INPUT <- path to input database folder | number of components for LBP & HOG
        OUTPUT -> accuracy & confusion matrix
    """

    def evaluate(self, in_db_path, nbr_cpnt_lbp, nbr_cpnt_hog):
        # print("-- EVALUATING --")
        feature_vectors = {"LBP": [], "HOG": []}
        labels = []

        for emotion in self.emotions:
            images = listdir(f"{in_db_path}/{emotion}/")

            for image in images:
                labels.append(emotion)
                in_img = self.preprocess(f"{in_db_path}/{emotion}/{image}")
                local_lbp_feat_vect, global_hog_feat_vect = self.describe(in_img)
                feature_vectors["LBP"].append(local_lbp_feat_vect)
                feature_vectors["HOG"].append(global_hog_feat_vect)

        skf = model_selection.StratifiedKFold(
            n_splits=10, shuffle=True, random_state=10
        )
        k_folds = skf.split(feature_vectors["LBP"], labels)

        accuracy_vect = []
        conf_mat_vect = []

        for train_index, test_index in k_folds:
            scaler_lbp, scaler_hog, pca_lbp, pca_hog, clf = self.train(
                np.array(feature_vectors["LBP"])[train_index],
                np.array(feature_vectors["HOG"])[train_index],
                np.array(labels)[train_index],
                nbr_cpnt_lbp,
                nbr_cpnt_hog,
                save=False,
            )
            accuracy, conf_mat = self.__test(
                np.array(feature_vectors["LBP"])[test_index],
                np.array(feature_vectors["HOG"])[test_index],
                np.array(labels)[test_index],
                scaler_lbp,
                pca_lbp,
                scaler_hog,
                pca_hog,
                clf,
            )

            accuracy_vect.append(accuracy)
            conf_mat_vect.append(conf_mat)

        accuracy = "%0.2f" % ((float(sum(accuracy_vect)) / len(accuracy_vect)) * 100)
        conf_mat = self.__mean_conf_mat(np.array(conf_mat_vect))

        print(f"Accuracy -> {accuracy}%")
        print("-> Confusion matrix <-")
        print(self.emotions)
        for idx in range(len(self.emotions)):
            print(conf_mat[idx])

    def __del__(self):
        print("-- End --")
