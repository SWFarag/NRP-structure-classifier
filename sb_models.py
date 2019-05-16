import pandas as pd
import numpy as np
from random import shuffle
import seaborn as sns
import argparse
import sys
import os

from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


import pickle
from rdkit.Chem.Draw import IPythonConsole #Needed to show molecules
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions #Only needed if modifying defaults

#### Importing all different descriptors and classifiers ####
from rdkit import Chem
from rdkit.Chem.EState import Fingerprinter
from rdkit.Chem import Descriptors
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Avalon.pyAvalonTools import GetAvalonFP #GetAvalonCountFP  #int vector version
from rdkit.Chem.AllChem import  GetMorganFingerprintAsBitVect, GetErGFingerprint
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem.rdMolDescriptors import *


import rdkit.DataStructs.cDataStructs
from rdkit.Chem import AllChem

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from scipy import interp
from itertools import cycle
import itertools
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score


from IPython.display import display

class MolecularModelling:

    def __init__(self, in_df_path=None, out_df_path=None, model_Type=None, descriptor_Type=None, algorithm_Type=None, scoring_metric=None):
        self.in_df_path      = in_df_path
        self.out_df_path     = out_df_path
        self.model_Type      = model_Type
        self.descriptor_Type = descriptor_Type
        self.algorithm_Type  = algorithm_Type
        self.scoring_metric  = scoring_metric


    def getData(self, path):
        df= pd.read_csv(path, index_col=0)
        print(df.shape)
        ## Add mol objects
        df['Mol'] = df['smiles'].apply(Chem.MolFromSmiles)
        for index, row in df.iterrows(): row['Mol'].SetProp("_Name", row['id'])
        print(df.shape)
        return df

    def splitData(self, df, path):
        ## don't forget to shuffle the data first

        x_train ,x_test = train_test_split(df,test_size=0.2, random_state=200, stratify=df['label'])

        x_train.to_csv(path +'trainingSet.csv')
        x_test.to_csv(path +'testSet.csv')
        x_train_w = Chem.SDWriter(path + 'trainingSet.sdf')
        x_test_w = Chem.SDWriter(path + 'testSet.sdf')

        for m in x_train['Mol']: x_train_w.write(m)
        for m in x_test['Mol']: x_test_w.write(m)
        print("Training_Set:", len(x_train))
        print("Test_Set:", len(x_test))
        y_train= x_train['label'].values
        y_test = x_test['label'].values
        x_train=x_train.drop(['label', 'Mol'], axis=1)
        x_test=x_test.drop(['label', 'Mol'], axis=1)
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        return  x_train ,x_test, y_train, y_test

    def computeDescriptors(self, data, descriptor_Type):
        y_data = data['label'].values
        print("type: ", type(descriptor_Type))
        if(isinstance(descriptor_Type, int)):
            fp_list = make_fingerprints(length=1024, verbose=True, mols=list(data['Mol']), chosen = descriptor_Type)
            data_descriptor = fp_list[0].x
            #st = StandardScaler()
            X_data = np.array(data_descriptor)
            print("X_data:", X_data.shape)
            X_data = X_data[~np.isnan(X_data).any(axis=1)]
            X_data = X_data[np.isfinite(X_data).any(axis=1)]
            print("dataSet_descripor before decorellation: ", X_data.shape)
            #X_data = self.removeHighlyCorrelatedFeatures(X_data, 0.95)
            print("dataSet_descripor after decorellation: ", X_data.shape)

            ###
            df = pd.DataFrame(X_data)
            print(df.shape)
            df['label'] = y_data
            print(df.shape)
            df['Mol'] = data['Mol'].values
            print(df.shape)

        else:
            df= pd.read_csv(descriptor_Type)
            print(df.shape)
            df = df[df.columns[2:]]
            print(df.shape)
            df['label'] = y_data
            print(df.shape)
            df['Mol'] = data['Mol'].values
            print(df.shape)

        return df

    def removeHighlyCorrelatedFeatures(self, X, cut_off):
        df = pd.DataFrame(X)
        # Create correlation matrix
        corr_matrix = df.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find index of feature columns with correlation greater than cut_off
        to_drop = [column for column in upper.columns if any(upper[column] > cut_off)]
        v = df.drop(df.columns[to_drop], axis=1)
        return np.array(v)


    def build_models_and_plot(self, X, y, model_dict, path, isplot, model_type, scoring_metric):
        ''' test a bunch of models and print out a sorted list of CV accuracies
            inputs:
                x: training data features, numpy array or Pandas dataframe
                y: training data labels, numpy array or Pandas dataframe
                model_dict: a dictionary of the form {name : model()}, where 'name' is a string
                            and 'model()' is a sci-kit-learn model object.
        '''
        n_folds=5
        random_state = np.random.RandomState(0)
        mean_aucs ={}
        mean_f1 = {}
         # Run classifier with cross-validation and plot ROC curves
        for (name, model) in model_dict.items():
            mean_aucs_list = []
            tprs = []
            aucs = []
            print("Model: ", name)
            cv = StratifiedKFold(n_splits=n_folds)
            i = 0
            scores = model_selection.cross_val_score(model, X, y, cv=n_folds, n_jobs=-1, scoring=scoring_metric)
            print("Scores: ", scores)
            plt.figure(figsize=(6, 4), dpi=150, facecolor='w', edgecolor='k')
            for train, test in cv.split(X, y):
                probas_ = model.fit(X[train], y[train] ).predict_proba(X[test])
                y_pred = model.fit(X[train], y[train] ).predict(X[test])
                if(scoring_metric=="balanced_accuracy"):
                    sm = balanced_accuracy_score(y[test], y_pred)
                else:
                    sm = f1_score(y[test], y_pred, average='macro')
                mean_aucs_list.append(self.auc_plotting(train, test, i, probas_, n_folds,
                                                       name, X,y, path, model_type, tprs,aucs))
                i += 1
                #print(set(y[test]), set(y_pred))
            self.save_model(name + "_" + model_type, path, model)
            mean_aucs[name]= [np.max(mean_aucs_list)]
            mean_f1 [name] = [np.mean(scores)]
        return mean_aucs, mean_f1


    def auc_plotting(self, train, test, foldnumber, probas_, n_folds, model_name, X, y, path, model_type, tprs, aucs):
        # Compute ROC curve and area the curve
        mean_auc = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (foldnumber, roc_auc))

        if(foldnumber==n_folds-1):
            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                     label='Chance', alpha=.8)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            print("mean_auc: ", mean_auc)
            std_auc = np.std(aucs)
            plt.plot(mean_fpr, mean_tpr, color='b',
                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                     lw=2, alpha=.8)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(model_name +' ROC')
            plt.legend(loc="lower right")
            #plt.show()
            plt.savefig(path + model_name + "_" + model_type +"_ROC.png", dpi=400, format='png')
            plt.clf()
        return mean_auc

    def predict_hidden_testSet(X_test, model_dict, path, class_names, model_type):
        predictions={}
        for (name, model) in model_dict.items():
            print(name)
            loaded_model=load_model(path + name + "_" + model_type + ".sav")
            y_pred = loaded_model.predict(X_test)
            predictions[name] = y_pred
        return predictions

    def predict_hidden_testSet_and_plot(self, X_test, y_test, model_dict, path, class_names, model_type):
         for (name, model) in model_dict.items():
            print(name)
            loaded_model = self.load_model(path + name + "_" + model_type + ".sav")
            y_pred = loaded_model.predict(X_test)
            self.compute_CM_and_stat(y_test, y_pred, class_names, path, name, model_type)
            print()


    def compute_CM_and_stat(self, y_test, y_pred, class_names, path, model_name, model_type):
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)
        # Plot non-normalized confusion matrix

        #figsize=(4, 4), dpi=80, facecolor='w', edgecolor='k'
        plt.figure(figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
        plt.title(model_name + " " + "Confusion Matrix")
        plt.subplot(121)
        self.plot_confusion_matrix(cnf_matrix, classes=class_names,
                              title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plt.subplot(122)
        self.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                              title='Normalized confusion matrix')
        plt.savefig(path + model_name + "_" + model_type +"_CM.png", dpi=400, format='png')
        plt.clf()

    def plot_confusion_matrix(self, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar(fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        #plt.show()

    def plot_confusion_matrix_stat(self, df):
        fig=plt.figure()
        ax1=plt.subplot(1, 2, 1)
        ax2=plt.subplot(1, 2, 2)
        fig.suptitle("Title for whole figure", fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        #sns.barplot(x='Types', y='Variances', data=df_final, ax=axs[0])
        ax1.set_title("Title for first plot")
        ax2.set_title("Title for first plot")
        sns.violinplot(x = df['mw'], data=df, ax=ax1,orient='v',color="green", gridsize=200)
        sns.violinplot(x = ['mw'], data=df,ax=ax2,orient='v')

    def save_model(self, model_name, path, model):
            filename = path + model_name +'.sav'
            print("filename: ", filename)
            pickle.dump(model, open(filename, 'wb'))

    def load_model(self, path):
        # load the model from disk
        return pickle.load(open(path, 'rb'))

    def setML(self, model_Type, algorithm_Type):
        model_dict = {}
        if (model_Type == 0):
            if   algorithm_Type == 1: model_dict['Random_Forest'] = RandomForestClassifier(n_estimators = 100)
            elif algorithm_Type == 2: model_dict['Logistic_Regression'] = LogisticRegression(solver='liblinear')
            elif algorithm_Type == 3: model_dict['SVM'] = SVC(gamma='auto', probability=True)
            elif algorithm_Type == 4: model_dict['Ensemble_Trees'] = Ensemble_Trees()
            elif algorithm_Type == "all":
                model_dict['Random_Forest'] = RandomForestClassifier(n_estimators=100)
                model_dict['Logistic_Regression'] = LogisticRegression(solver="liblinear")
                model_dict['SVM'] = SVC(gamma='auto', probability=True)
                #model_dict['Ensemble_Trees'] = Ensemble_Trees()
            else:
                 model_dict['Random_Forest'] = RandomForestClassifier()
        else:
            if   algorithm_Type == 1: model_dict['Random_Forest'] = RandomForestRegressor()
            elif algorithm_Type == 2: model_dict['Logistic_Regression'] = LinearRegression()
            elif algorithm_Type == 3: model_dict['SVM'] = SVR()
            elif algorithm_Type == 4: model_dict['Ensemble_Trees'] = GradientBoostingRegressor()
            elif algorithm_Type == "all":
                model_dict['Random_Forest'] = RandomForestRegressor()
                model_dict['Logistic_Regression'] = LinearRegression()
                model_dict['SVM'] = svr()
                model_dict['Ensemble_Trees'] = GradientBoostingRegressor()
            else:
                model_dict['Random_Forest'] = RandomForestClassifier()
        return model_dict

    def prepOutputFolder(self, out_df_path, model_Type, descriptor_Type, algorithm_Type):
            path = out_df_path
            try:
                os.mkdir(path)
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:
                print ("Successfully created the directory %s " % path)

            if(isinstance(descriptor_Type, int)):

                path = out_df_path + str(model_Type) +  "_" + str(descriptor_Type) +  "_" + str(algorithm_Type) + "/"

                try:
                    os.mkdir(path)
                except OSError:
                    print ("Creation of the directory %s failed" % path)
                else:
                    print ("Successfully created the directory %s " % path)
            else:

                path = out_df_path + str(model_Type) +  "_" + "Dragon" +  "_" + str(algorithm_Type) + "/"

                try:
                    os.mkdir(path)
                except OSError:
                    print ("Creation of the directory %s failed" % path)
                else:
                    print ("Successfully created the directory %s " % path)

            return path

    def ModelsSummary(self, df1, df2, scoring_metric):
        if(scoring_metric == "balanced_accuracy"):
            scoring_metric = "CCR"
        res1 = self.reshapeDf(df1, "AUC")
        res2 = self.reshapeDf(df2, scoring_metric)
        res1 = res1.sort_values(by=['AUC'], ascending=False)
        res2 = res2.sort_values(by=[scoring_metric], ascending=False)
        ###################################################
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(nrows=1,ncols=2,squeeze=False,sharex=False, sharey=True)
        fig.suptitle("Models Performance", fontsize=20)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.set_figheight(6)
        fig.set_figwidth(14)
        ax[0,0].set_title("AUC",fontsize=15)
        ax[0,1].set_title(scoring_metric, fontsize=15)
        #ax[0,0].set_xlabel(xlabel="fsadf",fontsize=24)
        #ax[0,1].set_xlabel(xlabel="fsadf",fontsize=24)
        #sns.set_context("paper", rc={"font.size":15,"axes.titlesize":10,"axes.labelsize":20})
        #sns.set()
        sns.set_context("paper",font_scale=1.6)
        sns.barplot(x="Model", y="AUC", hue="Descriptor", data=res1, ax = ax[0,0])
        sns.barplot(x="Model", y=scoring_metric, hue="Descriptor", data=res2, ax = ax[0,1])
        fig.savefig(fname=self.out_df_path + "SB_models_performance_summary_both.png" , dpi=400 ,format="png")
        fig.clf()
        return res1, res2

    def ModelsSummary1(self, df1):
        res1 = self.reshapeDf(df1, "AUC")
        res1 = res1.sort_values(by=['AUC'], ascending=False)
        sns.set_context("paper",font_scale=1.6)
        ax = sns.barplot(x="Model", y="AUC", data=res1)
        ax.set_title("AUC",fontsize=30)
        ax.figure.set_figheight(6)
        ax.figure.set_figwidth(8)
        ax.figure.savefig(fname=self.out_df_path + "SB_models_performance_summary_AUC.png" , dpi=400 ,format="png")
        ax.figure.clf()
        return res1

    def ModelsSummary2(self, df2, scoring_metric):
        if(scoring_metric == "balanced_accuracy"):
            scoring_metric = "CCR"
        res2 = self.reshapeDf(df2, scoring_metric)
        res2 = res2.sort_values(by=[scoring_metric], ascending=False)
        ###################################################
        #sns.set_context("paper",font_scale=1.6)
        ax = sns.barplot(x="Model", y=scoring_metric, data=res2)
        ax.set_title(scoring_metric, fontsize=20)
        ax.figure.set_figheight(7)
        ax.figure.set_figwidth(10)
        ax.figure.savefig(fname=self.out_df_path + "SB_models_performance_summary_" + scoring_metric +".png" , dpi=400 ,format="png")
        #ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        #plt.tight_layout()
        ax.figure.clf()
        return res2

    def reshapeDf(self, df, metric):
        y_axis = []
        x_axis = []
        des  = []
        dict_= {}
        for col in df.columns[0:-1]:
            y_axis.extend(list(df[col]))
            x_axis.extend([col]* len(df))

        des.extend(list(df['Descriptor'])*(len(df.columns)-1))
        dict_[metric] = y_axis
        dict_["Model"] = x_axis
        dict_["Descriptor"] = des
        res = pd.DataFrame.from_dict(dict_)
        return res

    def writeFinal_results_CCR(self, res, out_df_path):
        grouped = res.groupby(['Descriptor','Model'])
        grouped_df = grouped[['CCR']].agg(np.max)
        grouped_df_sorted=grouped_df.sort_values(['CCR'], ascending=[0])
        grouped_df_sorted
        grouped_df_sorted.to_csv(out_df_path + "SB_models_performance_summary_" + "CCR.csv")


    def writeFinal_results_AUC(self, res, out_df_path):
        grouped = res.groupby(['Descriptor', 'Model'])
        grouped_df = grouped[['AUC']].agg(np.max)
        grouped_df_sorted=grouped_df.sort_values(['AUC'], ascending=[0])
        grouped_df_sorted.to_csv(out_df_path + "SB_models_performance_summary_" + "AUC.csv")

    def runMM(self):
        print("Setting output Folder")
        out_path = self.prepOutputFolder(self.out_df_path, self.model_Type, self.descriptor_Type, self.algorithm_Type)
        print("done")
        print()
        print("Reading input file and creating molecules")
        df = self.getData(self.in_df_path)
        print("done")
        print()
        print("Computing descriptors")
        X_df = self.computeDescriptors(df, self.descriptor_Type)
        print("done")
        print()
        print("Splitting data into trainingset and test")
        x_train, x_test, y_train, y_test = self.splitData(X_df, out_path)

        print("type(x_train)", type(x_train))
        #display(x_train[0:2])
        print("done")
        print()
        isplot = True
        model_dict = self.setML(self.model_Type, self.algorithm_Type)
        print("Building models and running internal 5 FCV")
        mean_aucs_reg, mean_scoring_metric_reg = self.build_models_and_plot(x_train, y_train, model_dict,
                                                                            out_path, isplot, "regular", self.scoring_metric)
        print("mean_aucs_reg: ", mean_aucs_reg)
        print("mean_sm_reg: ", mean_scoring_metric_reg)
        print("done")
        print()
        print("Predict hidden test set")
        class_names=['Antibiotic', "Not_Antibiotic"]
        self.predict_hidden_testSet_and_plot(x_test, y_test, model_dict, out_path , class_names, "regular")
        print("done")
        print()
        print("Y-Randomized Models")
        print('############ Model Building #######')
        shuffle(y_train)
        mean_aucs_rand, mean_scoring_metric_rand = self.build_models_and_plot(x_train, y_train, model_dict,
                                                                              out_path, isplot, "randomized", self.scoring_metric)
        print("mean_aucs_rand: ", mean_aucs_rand)
        print("mean_sm_rand: ", mean_scoring_metric_rand)
        return mean_aucs_reg, mean_scoring_metric_reg


from rdkit.Chem import Descriptors
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem.rdMolDescriptors import *
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Avalon.pyAvalonTools import GetAvalonFP #GetAvalonCountFP  #int vector version
from rdkit.Chem.AllChem import  GetMorganFingerprintAsBitVect, GetErGFingerprint
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
import rdkit.DataStructs.cDataStructs
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import MACCSkeys

def ExplicitBitVect_to_NumpyArray(bitvector):
    bitstring = bitvector.ToBitString()
    intmap = map(int, bitstring)
    return np.array(list(intmap))


class fingerprint():
    def __init__(self, fp_fun, name):
        self.fp_fun = fp_fun
        self.name = name
        self.x = []

    def apply_fp(self, mols):
        for mol in mols:
            fp = self.fp_fun(mol)
            if isinstance(fp, tuple):
                fp = np.array(list(fp[0]))
            if isinstance(fp, rdkit.DataStructs.cDataStructs.ExplicitBitVect):
                fp = ExplicitBitVect_to_NumpyArray(fp)
            if isinstance(fp,rdkit.DataStructs.cDataStructs.IntSparseIntVect):
                fp = np.array(list(fp))

            self.x += [fp]

            if (str(type(self.x[0])) != "<class 'numpy.ndarray'>"):
                print("WARNING: type for ", self.name, "is ", type(self.x[0]))

def make_fingerprints(length, verbose, mols, chosen=None):
    if chosen   == 1:
        fp_list = [
            fingerprint(lambda x : GetHashedAtomPairFingerprintAsBitVect(x, nBits = length),
                     "&qfuot;Atom pair (1985)")]
    elif chosen == 2:
        fp_list = [
             fingerprint(lambda x : GetHashedTopologicalTorsionFingerprintAsBitVect(x, nBits = length),
                     "Topological torsion (1987)")]
    elif chosen == 3:
        fp_list = [
             fingerprint(lambda x : GetMorganFingerprintAsBitVect(x, 3, nBits = length),
                     "Morgan circular ")]
    elif chosen == 4:
         fp_list = [
             fingerprint(FingerprintMol, "Estate (1995)")]
    elif chosen == 5:
        fp_list = [
             fingerprint(lambda x: GetAvalonFP(x, nBits=length),
                    "Avalon bit based (2006)")]
    elif chosen == 6:
        fp_list = [
            fingerprint(lambda x: np.append(GetAvalonFP(x, nBits=length), Descriptors.MolWt(x)),
                    "Avalon+mol. weight")]
    elif chosen == 7:
        fp_list = [
            fingerprint(lambda x: GetErGFingerprint(x), "ErG fingerprint (2006)")]
    elif chosen == 8:
        fp_list = [
            fingerprint(lambda x : RDKFingerprint(x, fpSize=length),
                     "RDKit fingerprint")]
    elif chosen == 9:
        fp_list = [
            fingerprint(lambda x : FingerprintMols.FingerprintMol(x),
                     "RDKit fingerprint2")]
    else:
        fp_list = [fingerprint(lambda x : MACCSkeys.GenMACCSKeys(x), "RDKit MACCSkeys")]

    for fp in fp_list:
        if (verbose): print("doing", fp.name)
        fp.apply_fp(mols)

    return fp_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-in','--in_df_path',
                        help='Path of input file path.')
    parser.add_argument('-o', '--out_df_path',
                        help='Path to new .csv file for saving the potential peptide dataframe.')
    parser.add_argument('-mt', '--model_Type', type=int,
                        help='determine whether it is a continiuous or a classification model.')
    parser.add_argument('-dt', '--descriptor_Type',
                        help='determine which descriptor type to be used.')
    parser.add_argument('-at', '--algorithm_Type',
                        help='determine which machine learning algorithm.')
    parser.add_argument('-sm', '--scoring_metric',
                        help='determined the scoring metic e.g: balanced_accuracy, f1-score.')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    d = args.descriptor_Type
    mean_aucs_reg_dfs = []
    mean_sm_reg_dfs = []
    for i in d.split(","):
        if (i.isdigit()):
            descriptor_Type = int(i)
        else:
            descriptor_Type = i
            print(i.isdigit(), i)
        mm = MolecularModelling(in_df_path      = args.in_df_path,
                                out_df_path     = args.out_df_path,
                                model_Type      = args.model_Type,
                                descriptor_Type = descriptor_Type,
                                algorithm_Type  = args.algorithm_Type,
                                scoring_metric  = args.scoring_metric)

        mean_aucs_reg, mean_sm_reg = mm.runMM()
        mean_aucs_reg_df = pd.DataFrame.from_dict(mean_aucs_reg)
        mean_sm_reg_df = pd.DataFrame.from_dict(mean_sm_reg)

        if(i.isdigit()):
            mean_aucs_reg_df['Descriptor'] = [i]
            mean_sm_reg_df['Descriptor'] = [i]
        else:
            mean_aucs_reg_df['Descriptor'] = ["Dragon"]
            mean_sm_reg_df['Descriptor'] = ["Dragon"]
    mean_aucs_reg_dfs.append(mean_aucs_reg_df)
    mean_sm_reg_dfs.append(mean_sm_reg_df)
mean_aucs_reg_dfs_final = pd.concat(mean_aucs_reg_dfs)
mean_sm_reg_dfs_final = pd.concat(mean_sm_reg_dfs)

res1, res2 = mm.ModelsSummary(mean_aucs_reg_dfs_final, mean_sm_reg_dfs_final, args.scoring_metric)
res1 = mm.ModelsSummary1(mean_aucs_reg_dfs_final)
res2 = mm.ModelsSummary2(mean_sm_reg_dfs_final, args.scoring_metric)
mm.writeFinal_results_AUC(res1, args.out_df_path)
mm.writeFinal_results_CCR(res2, args.out_df_path)
