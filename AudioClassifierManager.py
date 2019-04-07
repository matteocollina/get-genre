'''
AudioClassifierManager is used to manage classification
'''
from pyAudioAnalysis import audioBasicIO as aBIO
from pyAudioAnalysis import audioFeatureExtraction as aF
from pyAudioAnalysis import audioTrainTest as aT
import numpy
import utils
import pickle as cPickle

class AudioClassifierManager:
    # experiment executed per parameter in classification
    __num_experiment = 100
    __compute_beat = True
    __svmModelName = "svm"
    __svmRbfModelName = "svm_rbf"
    __randomforestModelName = "randomforest"
    __knnModelName = "knn"
    __gradientboostingModelName = "gradientboosting"
    __extratreesModelName = "extratrees"
    __models = ["knn", "svm","svm_rbf", "randomforest","gradientboosting"]
    __perTrain = [0.9, 0.7]

    BEST_ACCURACY = 0
    BEST_F1 = 1

    @staticmethod
    def getAllModels():
        return AudioClassifierManager.__models

    @staticmethod
    def getModelNameForTypeAndPt(model_type,pT):
        return "{0}_pt{1}".format(model_type, str(int(pT * 100)))

    @staticmethod
    def getPerTrainProportions():
        return AudioClassifierManager.__perTrain

    @staticmethod
    def getMtWin():
        return aT.shortTermWindow * 2

    @staticmethod
    def getMtStep():
        return aT.shortTermStep * 2

    @staticmethod
    def getStWin():
        return aT.shortTermWindow

    @staticmethod
    def getStStep():
        return aT.shortTermStep

    @staticmethod
    def getFeaturesAndClasses(dirs):
        return aF.dirsWavFeatureExtraction(dirs,
                                            AudioClassifierManager.getMtWin(),
                                            AudioClassifierManager.getMtStep(),
                                            AudioClassifierManager.getStWin(),
                                            AudioClassifierManager.getStStep(),
                                            compute_beat=AudioClassifierManager.__compute_beat)

    @staticmethod
    def getFeaturesOptimized(features):
        '''
        :param features: global matrix
        :return: matrix optimized without feature vectors with NaN or Inf
        '''
        featuresOpt = []
        for matFeatClass in features:
            fTemp = []
            for i in range(matFeatClass.shape[0]):
                temp = matFeatClass[i, :]
                # Verify if current vector of features of track has Nan or inf
                if (not numpy.isnan(temp).any()) and (not numpy.isinf(temp).any()):
                    fTemp.append(temp.tolist())
                else:
                    print("NaN Found! Feature vector not used for training")
            featuresOpt.append(numpy.array(fTemp))
        return featuresOpt

    @staticmethod
    def getCountClasses(features):
        '''
        :param features: matrix of features
        :return: number of classes (genres)
        '''
        return len(features)

    @staticmethod
    def getCountTracksInClass(matrix_feature_class):
        '''
        :param matrix_feature_class: matrix of features of a class
        :return: number of tracks for genre
        '''
        return len(matrix_feature_class)

    @staticmethod
    def getCountFeatures(features):
        '''
        :param features: global matrix with classes and vector of features inside
        :return: number of features ('zcr', 'energy', 'energy_entropy', 'spectral_centroid' , e.g)
        '''
        # get columns of first genre of matrix
        return features[0].shape[1]

    @staticmethod
    def writeTrainDataToARFF(model_name, features, classNames):
        n_feats = AudioClassifierManager.getCountFeatures(features)
        feature_names = ["features" + str(d + 1) for d in range(n_feats)]
        aT.writeTrainDataToARFF(model_name, features, classNames, feature_names)


    @staticmethod
    def getListParamsForClassifierType(classifier_type):
        if classifier_type == AudioClassifierManager.__svmModelName or classifier_type == AudioClassifierManager.__svmRbfModelName:
            classifier_par = numpy.array([0.001, 0.01, 0.5, 1.0, 5.0, 10.0, 20.0])
        elif classifier_type == AudioClassifierManager.__randomforestModelName:
            classifier_par = numpy.array([10, 25, 50, 100, 200, 500])
        elif classifier_type == AudioClassifierManager.__knnModelName:
            classifier_par = numpy.array([1, 3, 5, 7, 9, 11, 13, 15])
        elif classifier_type == AudioClassifierManager.__gradientboostingModelName:
            classifier_par = numpy.array([10, 25, 50, 100, 200, 500])
        elif classifier_type == AudioClassifierManager.__extratreesModelName:
            classifier_par = numpy.array([10, 25, 50, 100, 200, 500])
        else:
            classifier_par  = numpy.array([])
        return classifier_par

    @staticmethod
    def getOptimalNumberExperiment(features,n_exp):
        '''
        :param features: global matrix
        :param n_exp: number of experiment from user
        :return:
        '''
        n_samples_total = 0
        for f in features:
            n_samples_total += f.shape[0]
        print("Number of total samples: {0}".format(n_samples_total))
        if n_samples_total > 1000 and n_exp > 50:
            n_exp = 50
            print("Number of training experiments changed to 50 due to high number of samples")
        if n_samples_total > 2000 and n_exp > 10:
            n_exp = 10
            print("Number of training experiments changed to 10 due to high number of samples")
        return n_exp

    @staticmethod
    def getTrainClassifier(f_train,classifier_name,param):
        if classifier_name == AudioClassifierManager.__svmModelName:
            classifier = aT.trainSVM(f_train, param)
        elif classifier_name == AudioClassifierManager.__svmRbfModelName:
            classifier = aT.trainSVM_RBF(f_train, param)
        elif classifier_name == AudioClassifierManager.__knnModelName:
            classifier = aT.trainKNN(f_train, param)
        elif classifier_name == AudioClassifierManager.__randomforestModelName:
            classifier = aT.trainRandomForest(f_train, param)
        elif classifier_name == AudioClassifierManager.__gradientboostingModelName:
            classifier = aT.trainGradientBoosting(f_train, param)
        elif classifier_name == AudioClassifierManager.__extratreesModelName:
            classifier = aT.trainExtraTrees(f_train, param)
        else:
            classifier = None
        return classifier

    @staticmethod
    def saveClassifierModel(features,model_name,classifier_type,classifier,MEAN,STD,classNames,bestParam):
        if classifier_type == "knn":
            [X, Y] = aT.listOfFeatures2Matrix(features)
            X = X.tolist()
            Y = Y.tolist()
            fo = open(model_name, "wb")
            cPickle.dump(X, fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(Y, fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(MEAN, fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(STD, fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(classNames, fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(bestParam, fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(AudioClassifierManager.getMtWin(), fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(AudioClassifierManager.getMtStep(), fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(AudioClassifierManager.getStWin(), fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(AudioClassifierManager.getStStep(), fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(AudioClassifierManager.__compute_beat, fo, protocol=cPickle.HIGHEST_PROTOCOL)
            fo.close()
        elif classifier_type == AudioClassifierManager.__svmModelName or classifier_type == AudioClassifierManager.__svmRbfModelName or \
                        classifier_type == AudioClassifierManager.__randomforestModelName or \
                        classifier_type == AudioClassifierManager.__gradientboostingModelName or \
                        classifier_type == AudioClassifierManager.__extratreesModelName:
            with open(model_name, 'wb') as fid:
                cPickle.dump(classifier, fid)
            fo = open(model_name + "MEANS", "wb")
            cPickle.dump(MEAN, fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(STD, fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(classNames, fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(AudioClassifierManager.getMtWin(), fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(AudioClassifierManager.getMtStep(), fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(AudioClassifierManager.getStWin(), fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(AudioClassifierManager.getStStep(), fo, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(AudioClassifierManager.__compute_beat, fo, protocol=cPickle.HIGHEST_PROTOCOL)
            fo.close()

    @staticmethod
    def getResultMatrixAndBestParam(features, class_names, classifier_name, parameterMode, perTrain=0.90, model_name='',Params=[]):
        '''
        ARGUMENTS:
            features:     a list ([numOfClasses x 1]) whose elements containt numpy matrices of features.
                    each matrix features[i] of class i is [n_samples x numOfDimensions]
            class_names:    list of class names (strings)
            n_exp:        number of cross-validation experiments
            classifier_name: svm or knn or randomforest
            Params:        list of classifier parameters (for parameter tuning during cross-validation)
            parameterMode:    0: choose parameters that lead to maximum overall classification ACCURACY
                    1: choose parameters that lead to maximum overall f1 MEASURE
        RETURNS:
             bestParam:    the value of the input parameter that optimizes the selected performance measure
             confufionMatrix

        '''
        # feature normalization:
        (features_norm, MEAN, STD) = aT.normalizeFeatures(features)

        n_classes = len(features)
        ac_all = []
        f1_all = []
        precision_classes_all = []
        recall_classes_all = []
        f1_classes_all = []
        cms_all = []
        smooth = 0.0000000010

        # Optimize number of experiment
        n_exp = AudioClassifierManager.getOptimalNumberExperiment(features,AudioClassifierManager.__num_experiment)

        Params = AudioClassifierManager.getListParamsForClassifierType(classifier_name) if len(Params)==0 else Params

        # For each param value
        for Ci, C in enumerate(Params):
            # Init confusion matrix
            cm = numpy.zeros((n_classes, n_classes))
            for e in range(n_exp):
                # Split features in Train and Test:
                f_train, f_test = aT.randSplitFeatures(features_norm, perTrain)
                countFTrain = 0
                countFTest = 0
                for g in f_train:
                    for track in g:
                        countFTrain += 1
                for g in f_test:
                    for track in g:
                        countFTest += 1

                if(countFTest == 0):
                    print("WARNING: {0} has no test values".format(class_names[Ci]))

                # for each cross-validation iteration:
                print("Param = {0:.5f} - classifier Evaluation "
                      "Experiment {1:d} of {2:d} - lenTrainingSet {3} lenTestSet {4}".format(C, e + 1, n_exp,
                                                                                                        countFTrain,
                                                                                                        countFTest))

                # Get Classifier for train
                classifier = AudioClassifierManager.getTrainClassifier(f_train,classifier_name,C)


                cmt = numpy.zeros((n_classes, n_classes))
                for c1 in range(n_classes):
                    #print("==> Class {1}: {0} for exp {2}".format(class_names[c1],c1,e))
                    n_test_samples = len(f_test[c1])
                    res = numpy.zeros((n_test_samples, 1))
                    for ss in range(n_test_samples):
                        [res[ss], _] = aT.classifierWrapper(classifier,
                                                         classifier_name,
                                                         f_test[c1][ss])
                    for c2 in range(n_classes):
                        nnzero = numpy.nonzero(res == c2)[0]
                        rlen = len(nnzero)
                        cmt[c1][c2] = float(rlen)
                        #print("cmt[{0}][{1}] = {2}".format(c1,c2,float(rlen)))
                cm = cm + cmt


            cm = cm + smooth
            rec = numpy.zeros((cm.shape[0],))
            pre = numpy.zeros((cm.shape[0],))

            # Calculate Precision, Recall and f1 Misure
            for ci in range(cm.shape[0]):
                rec[ci] = cm[ci, ci] / numpy.sum(cm[ci, :])
                pre[ci] = cm[ci, ci] / numpy.sum(cm[:, ci])
            precision_classes_all.append(pre)
            recall_classes_all.append(rec)
            f1 = 2 * rec * pre / (rec + pre)
            f1_classes_all.append(f1)
            ac_all.append(numpy.sum(numpy.diagonal(cm)) / numpy.sum(cm))

            cms_all.append(cm)
            f1_all.append(numpy.mean(f1))


        best_ac_ind = numpy.argmax(ac_all)
        best_f1_ind = numpy.argmax(f1_all)
        bestParam = 0
        resultConfusionMatrix = None
        if parameterMode == AudioClassifierManager.BEST_ACCURACY:
            bestParam = Params[best_ac_ind]
            resultConfusionMatrix = cms_all[best_ac_ind]
        elif parameterMode == AudioClassifierManager.BEST_F1:
            bestParam = Params[best_f1_ind]
            resultConfusionMatrix = cms_all[best_f1_ind]

        return bestParam, resultConfusionMatrix, precision_classes_all, recall_classes_all, f1_classes_all, f1_all, ac_all

    @staticmethod
    def saveConfusionMatrix(cm, class_names, classifier_name='ns'):
        '''
        This function prints a confusion matrix for a particular classification task.
        ARGUMENTS:
            cm:            a 2-D numpy array of the confusion matrix
                           (cm[i,j] is the number of times a sample from class i was classified in class j)
            class_names:    a list that contains the names of the classes
        '''
        cmCsv = numpy.empty([len(class_names), len(class_names)])
        header = []  # header of matrix
        header.append("/")

        if cm.shape[0] != len(class_names):
            print("printConfusionMatrix: Wrong argument sizes\n")
            return

        for i, c in enumerate(class_names):
            header.append(c)

        for i, c in enumerate(class_names):
            for j in range(len(class_names)):
                val = 100.0 * cm[i][j] / numpy.sum(cm)
                cmCsv[i][j] = format(val, '.2f')

        # Save as csv
        out = numpy.column_stack([class_names, cmCsv])
        out1 = numpy.row_stack([header, out])
        numpy.savetxt('confusion_matrix_{0}.csv'.format(classifier_name), out1, delimiter=',', fmt="%s")
        utils.save_plot(utils.get_confusion_matrix(cmCsv, class_names, class_names),
                        "confusion_matrix_{0}".format(classifier_name))

    @staticmethod
    def saveParamsFromClassification(class_names,Params,model_name,precision_classes_all,recall_classes_all,f1_classes_all,ac_all,f1_all):
        '''
        :param class_names: Name of classes
        :param Params: Params for model type
        :param model_name: Name of model of classification
        :param precision_classes_all: list of precisions
        :param recall_classes_all: list of recalls
        :param f1_classes_all: list of f1
        :param ac_all: accuracy for param
        :param f1_all: mean f1 for param
        '''
        header = []
        headerParams = []
        matrixValues = numpy.zeros([len(Params), (len(class_names) * 3) + 5])
        header.append("/")

        for i, c in enumerate(class_names):
            header.extend([c, "", ""])

        header.extend(["OVERALL", "", "", ""])
        headerParams.append("C")

        for c in class_names:
            headerParams.extend(["PRE", "REC", "f1"])

        headerParams.extend(["ACC", "f1", "best ACC", "best f1"])

        best_ac_ind = numpy.argmax(ac_all)
        best_f1_ind = numpy.argmax(f1_all)

        for i in range(len(precision_classes_all)):
            matrixValues[i][0] = Params[i]

            for c in range(len(precision_classes_all[i])):
                curr_pr = 100.0 * precision_classes_all[i][c]
                curr_rec = 100.0 * recall_classes_all[i][c]
                curr_f1 = 100.0 * f1_classes_all[i][c]
                matrixValues[i][(c * 3) + 1] = curr_pr
                matrixValues[i][(c * 3) + 2] = curr_rec
                matrixValues[i][(c * 3) + 3] = curr_f1

            curr_acc_all = 100.0 * ac_all[i]
            curr_f1_all = 100.0 * f1_all[i]
            matrixValues[i][(len(class_names) * 3) + 1] = curr_acc_all
            matrixValues[i][(len(class_names) * 3) + 2] = curr_f1_all

            if i == best_f1_ind:
                matrixValues[i][(len(class_names) * 3) + 4] = 1
            if i == best_ac_ind:
                matrixValues[i][(len(class_names) * 3) + 3] = 1

        out = numpy.row_stack([headerParams, matrixValues])
        out1 = numpy.row_stack([header, out])
        numpy.savetxt('params_{0}.csv'.format(model_name), out1, delimiter=',', fmt="%s")