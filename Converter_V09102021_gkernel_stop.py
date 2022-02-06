# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:43:42 2019

@author: solale
In this version I will try to shrink the network and reduce the tensorization
"""
# Multilayer Perceptron
import pandas
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# from tensorflow import set_random_seed
# set_random_seed(2)

import tensorflow
import tensorflow.keras
import math
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose,Input, Reshape, Conv2D, Flatten
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
# from tensorflow.keras.layers.merge import concatenate
from tensorflow.keras.layers import concatenate
import argparse
# from tensorflow.keras.utils.np_utils import to_categorical
from tensorflow.keras import utils
# import tensorflow as tf
from sklearn import preprocessing


from keras_ex.gkernel import GaussianKernel
# https://github.com/darecophoenixx/wordroid.sblo.jp/tree/master/lib/keras_ex/gkernel


def custom_loss_1 (y_true, y_pred):
    A = tensorflow.keras.losses.mean_squared_error(y_true[:,0:4], y_pred[:,0:4])
    return A

def custom_loss_2 (y_true, y_pred):
    B =  tensorflow.keras.losses.categorical_crossentropy(y_true[:,-4:], y_pred[:,-4:])
    return n*B

def custom_loss (y_true, y_pred):
    A =  tensorflow.keras.losses.mean_squared_error(y_true[:,0:4], y_pred[:,0:4])
    B =  tensorflow.keras.losses.categorical_crossentropy(y_true[:,-4:], y_pred[:,-4:])
    m=1
    return((m*A)+ (n*B))

########################## argument getting

#parser = argparse.ArgumentParser()
#parser.add_argument("--i",  )
#parser.add_argument("--j", )
#parser.add_argument("--k", )
#parser.add_argument("--m", )
#a = parser.parse_args()
##
#i=int(a.i)
#j=int(a.j)
#k=int(a.k)

#####
i = 64 #64
# j=16
# k=64

n = 75 #105

n_splits=10
max_epochs=500
BatchSize=350
N_AlternatingControler=2

######################  ######################
# Reading Multi Modal Y for train and Test
# train data
AllDataset = pandas.read_csv('./XY_BLD_Converter', low_memory=False)
AllDataset = AllDataset.set_index(AllDataset.RID)
AllDataset = AllDataset.fillna(0)
AllDataset['DX'] = AllDataset['DX'].map({'NL':0, 'MCI':1, 'Converter':2, 'Dementia':3})
le = preprocessing.LabelEncoder()
AllDataset['DX'] = le.fit_transform(AllDataset['DX'])

###################### MRI ######################
MRI_X = AllDataset.loc[:,['Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV']]
MRI_Y = AllDataset.loc[:, ['MMSE_BLD','MMSE_6','MMSE_12','MMSE_24','DX']]
MRI_RID = AllDataset.RID
# normalize data
MRI_X = (MRI_X - MRI_X.mean())/ (MRI_X.max() - MRI_X.min())

###################### PET ######################
PET_X = AllDataset.loc[:,['FDG', 'PIB', 'AV45']]
PET_Y = AllDataset.loc[:, ['MMSE_BLD','MMSE_6','MMSE_12','MMSE_24','DX']]
PET_RID = AllDataset.RID
# normalize data
PET_X = (PET_X - PET_X.mean()) / (PET_X.max() - PET_X.min())

###################### COG ######################
COG_X = AllDataset.loc[:, ['RAVLTimmediate', 'RAVLTlearning', 'RAVLTforgetting', 'RAVLTpercforgetting','FAQ', 
                'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan', 'EcogPtDivatt', 'EcogPtTotal',
                'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat', 'EcogSPPlan', 'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal']]#'CDRSB', 'MOCA',
COG_Y = AllDataset.loc[:, ['MMSE_BLD','MMSE_6','MMSE_12','MMSE_24','DX']]
COG_RID = AllDataset.RID
# normalize data
COG_X = (COG_X - COG_X.mean()) / (COG_X.std())

###################### CSF ######################
CSF_X = AllDataset.loc[:,['ABETA', 'PTAU', 'TAU']]
CSF_Y = AllDataset.loc[:, ['MMSE_BLD','MMSE_6','MMSE_12','MMSE_24','DX']]
CSF_RID = AllDataset.RID
# normalize data
CSF_X = (CSF_X - CSF_X.mean()) / (CSF_X.max() - CSF_X.min())

###################### RF ######################
# RF_X = AllDataset.loc[:,['AGE', 'PTEDUCAT', 'APOE4','female','male']]
# RF_Y = AllDataset.loc[:, ['MMSE_BLD','MMSE_6','MMSE_12','MMSE_24','DX']]
# RF_RID = AllDataset.RID
# # normalize data
# RF_X.AGE = (RF_X - RF_X.mean()) / (RF_X.max() - )

RF_X_1 = AllDataset.loc[:,['AGE','PTEDUCAT']] 
# normalize age and years of education
RF_X_1 = (RF_X_1 - RF_X_1.mean()) / (RF_X_1.max() - RF_X_1.min())
RF_X_1=RF_X_1.fillna(0)
# normalize apoe4
RF_X_A = AllDataset.loc[:,['APOE4']]
RF_X_A=RF_X_A-1
RF_X_A=RF_X_A.fillna(0)
# normalize gender
RF_X_gender = AllDataset.loc[:,['female','male']] 
# RF_X_sex[RF_X_sex=='Male']=-1
# RF_X_sex[RF_X_sex=='Female']=1
RF_X_gender=RF_X_gender.fillna(0)
#construct RF 
RF_X = pandas.concat([RF_X_1, RF_X_A, RF_X_gender], axis=1)

##############################################

from tensorflow.keras.layers import Dropout
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr, spearmanr

#  FCN specifications
units_L2 = 25
units_L3 = 7

####################################### MRI FCN ###############################################
# mri FCN
MRI_inp_dim = MRI_X.shape[1]
MRI_visible = Input(shape=(MRI_inp_dim,))
hiddenMRI1 = Dense(2*MRI_inp_dim, kernel_initializer='normal', activation='linear')(MRI_visible)
hiddenMRI2 = hiddenMRI1
MRI_output = Dense(MRI_inp_dim, kernel_initializer='normal', activation='linear')(hiddenMRI2)

####################################### PET FCN ###############################################
PET_inp_dim = PET_X.shape[1]
PET_visible = Input(shape=(PET_inp_dim,))
hiddenPET1 = Dense(2*PET_inp_dim, kernel_initializer='normal', activation='linear')(PET_visible)
hiddenPET2=hiddenPET1
PET_output = Dense(PET_inp_dim, kernel_initializer='normal', activation='linear')(hiddenPET2)

####################################### COG FCN ###############################################
# mri FCN
COG_inp_dim = COG_X.shape[1]
COG_visible = Input(shape=(COG_inp_dim,))
hiddenCOG1 = Dense(2*COG_inp_dim, kernel_initializer='normal', activation='linear')(COG_visible)
hiddenCOG2=hiddenCOG1
COG_output = Dense(COG_inp_dim, kernel_initializer='normal', activation='linear')(hiddenCOG2)

####################################### CSF FCN ###############################################
CSF_inp_dim = CSF_X.shape[1]
CSF_visible = Input(shape=(CSF_inp_dim,))
hiddenCSF1 = Dense(2*CSF_inp_dim, kernel_initializer='normal', activation='linear')(CSF_visible)
hiddenCSF2=hiddenCSF1
CSF_output = Dense(CSF_inp_dim, kernel_initializer='normal', activation='linear')(hiddenCSF2)
####################################### CSF FCN ###############################################
RF_inp_dim = RF_X.shape[1]
RF_visible = Input(shape=(RF_inp_dim,))
hiddenRF1 = Dense(2*RF_inp_dim, kernel_initializer='normal', activation='linear')(RF_visible)
hiddenRF2=hiddenRF1
RF_output = Dense(RF_inp_dim, kernel_initializer='normal', activation='linear')(hiddenRF2)

#################################### Concat FCN ###############################################

merge = concatenate([MRI_output, PET_output, COG_output, CSF_output, RF_output])#
# print(merge.shape[1])
# interpretation layer
# hidden1 = Dense(100, activation='relu')(merge)
hidden1 = GaussianKernel(100, merge.shape[1], kernel_gamma="auto", name='gkernel1')(merge)

# hidden1 = Dropout(0.1)(hidden1)

hidden1_reshape = Reshape((10, 10, 1))(hidden1)
layer2D_1 = Conv2DTranspose(filters=10, kernel_size=(3,3), strides=(1, 1), padding="same")(hidden1_reshape)
layer2D_2 = Conv2DTranspose(filters=10, kernel_size=(3,3), strides=(1, 1), dilation_rate=(2,2),padding="same")(hidden1_reshape)
#layer2D_3 = Conv2DTranspose(filters=10, kernel_size=(3,3), strides=(1, 1), dilation_rate=(3,3), padding="same")(hidden1_reshape)

layer2D_4 = concatenate([layer2D_1,layer2D_2])#concatenate([layer2D_1,layer2D_2,layer2D_3])

    
# input layer
visible = layer2D_4

# first feature extractor
conv1 = Conv2D(i, kernel_size=3)(visible)#relu

conv1 = Dropout(0.1)(conv1)

flat1 = Flatten()(conv1)
## cutting out from hidden1 output
# prediction output
output_reg = Dense(4, activation='relu',kernel_regularizer=tensorflow.keras.regularizers.l1(0.01))(flat1)#relu
outout_class = Dense(4, activation='softmax',kernel_regularizer=tensorflow.keras.regularizers.l1(0.01))(flat1)#softmax

output=concatenate([output_reg, outout_class])

categorical_labels = utils.to_categorical(COG_Y.iloc[:,-1], num_classes=4)

X_all=[MRI_X.values, PET_X.values, COG_X.values, CSF_X.values, RF_X.values]# 
YTrain = COG_Y
YTrain1 = YTrain.reset_index()
Y_Train = pandas.concat ([YTrain1[['MMSE_BLD','MMSE_6','MMSE_12','MMSE_24']], pandas.DataFrame(categorical_labels)], axis=1)
Y_all=Y_Train


AccScores = []  
AccDetails=[]
All_Predicts_class=[]
All_Truth_class=[]
All_Predicts_reg=[]
All_Truth_reg=[]
AllRegErrors = np.zeros(shape=(4,1),dtype='float16')
X_all=[MRI_X.values, PET_X.values, COG_X.values, CSF_X.values, RF_X.values]#
Y_all=Y_Train
All_RMSE=np.zeros(shape=(4,1),dtype='float16')


model = Model(inputs= [MRI_visible, PET_visible, COG_visible, CSF_visible, RF_visible], outputs=output)    #

#keras.utils.plot_model(model,to_file='model-final.png', show_shapes=True)

OPTIMIZER_1=tensorflow.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
OPTIMIZER_2=tensorflow.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.save_weights('SavedInitialWeights.h5')

callback_stop =  tensorflow.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=20, verbose=0,
    mode='auto', baseline=None, restore_best_weights=False
)


max_epochs_Alternating=max_epochs/N_AlternatingControler
max_epochs_Alternating=np.int(max_epochs_Alternating)

import matplotlib.pyplot as plt




for repeator in range(0,1):
    #print('Repeat No:  ', repeator+1)
    # define  n_splits-fold cross validation test harness

    kfold = StratifiedKFold(n_splits, shuffle=True, random_state=repeator)
    FoldCounter=0
    for train, test in kfold.split(X_all[1], COG_Y.iloc[:,-1].values):
        FoldCounter=FoldCounter+1        

        model.load_weights('SavedInitialWeights.h5')        
        
        X_train_here=[X_all[0][train], X_all[1][train], X_all[2][train],  X_all[3][train], X_all[4][train]]#
        print('---Repeat No:  ', repeator+1, '  ---Fold No:  ', FoldCounter)        
        
        # model.compile(loss=custom_loss, optimizer=OPTIMIZER_1)
        # History = model.fit(X_train_here, Y_all.values[train], 
        #                     epochs= max_epochs_Alternating, batch_size=BatchSize, verbose=0)#250-250
        model.compile(loss=custom_loss, optimizer=OPTIMIZER_2)
        History = model.fit(X_train_here, Y_all.values[train], validation_split=0.1,
                            epochs= 2*max_epochs_Alternating, batch_size=BatchSize,
                            callbacks=[callback_stop], verbose=0)#250-250
        
        plt.plot(History.history['loss'])
        plt.plot(History.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.grid()
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('Fold_'+str(FoldCounter)+'_History.png')
        plt.close()
        
#        for iters in range(N_AlternatingControler):        
#            # Fit the model
#            if np.random.rand() < 0.5:                
#                model.compile(loss=custom_loss, optimizer=OPTIMIZER_1)
#                print(iters)
#            else:
#                model.compile(loss=custom_loss, optimizer=OPTIMIZER_2)
#            History = model.fit(X_train_here, Y_all.values[train], epochs= max_epochs_Alternating, batch_size=BatchSize, verbose=0)#250-250
        
        X_test_here=[X_all[0][test], X_all[1][test], X_all[2][test], X_all[3][test], X_all[4][test]]#
        Y_Validation=model.predict(X_test_here)
        MSE_0 = mean_squared_error(Y_all.iloc[test, 0], Y_Validation[:, 0])#/Y_Pred_MultiModal.shape[0]
        MSE_6 = mean_squared_error(Y_all.iloc[test, 1], Y_Validation[:, 1])#/Y_Pred_MultiModal.shape[0]
        MSE_12 = mean_squared_error(Y_all.iloc[test, 2], Y_Validation[:, 2])#/Y_Pred_MultiModal.shape[0]
        MSE_24 = mean_squared_error(Y_all.iloc[test, 3], Y_Validation[:, 3])#/Y_Pred_MultiModal.shape[0]
        All_RMSE[0]=math.sqrt(MSE_0)
        All_RMSE[1]=math.sqrt(MSE_6)
        All_RMSE[2]=math.sqrt(MSE_12)
        All_RMSE[3]=math.sqrt(MSE_24)
        print([math.sqrt(MSE_0), math.sqrt(MSE_6), math.sqrt(MSE_12), math.sqrt(MSE_24)])
        AllRegErrors=np.append(AllRegErrors,All_RMSE,axis=1)
        
        #rho1, pval1 =  spearmanr(Y_Pred_MultiModal[:, 0], Y_all.iloc[:, 0])
        ##### Classification
        All_Predicts_class.append(Y_Validation[:,-4:])
        All_Predicts_reg.append(Y_Validation[:,0:4])
        All_Truth_class.append(COG_Y.iloc[test,-1])
        All_Truth_reg.append(Y_all.iloc[test, 0:4])
        DX_pred = np.argmax(Y_Validation[:,-4:], axis=1)    
        DX_real= COG_Y.iloc[test,-1]
        score=accuracy_score(DX_real, DX_pred)
        print (accuracy_score(DX_real, DX_pred))
        AccScores.append(score*100)
        
        # target_names = ['class 0', 'class 1', 'class 2', 'class 3']
        target_names = ['CN', 'MCI_nc',  'MCI_c', 'AD']
        class_names = target_names
        Details=classification_report(DX_real, DX_pred, target_names=target_names,output_dict=True)
        print(classification_report(DX_real, DX_pred, target_names=target_names))
        AccDetails.append(Details)
        
        #print >> f1, classification_report(DX_real, DX_pred, target_names=target_names)           

print('#########################################################################')
print('#########################################################################')
print(i, n)
print('Average Result:')      
print('########')
print('Mean of RMSE :   ', np.mean(AllRegErrors[:,1:],1))
print('Mean of RMSE ALL:  ', np.mean(AllRegErrors[:,1:]))
print('Mean of accuracy :  ',np.mean(AccScores))
print('  ---------------------  ')
print('std of RMSE :   ', np.std(AllRegErrors[:,1:],1))
print('std of RMSE ALL:  ', np.std(AllRegErrors[:,1:]))
print('std of accuracy :  ',np.std(AccScores))


AD_precision=[]; MCI_nc_precision=[]; MCI_c_precision=[]; CN_precision=[];
AD_recall=[]; MCI_nc_recall=[]; MCI_c_recall=[]; CN_recall=[]
AD_f1=[]; MCI_nc_f1=[]; MCI_c_f1=[]; CN_f1=[]
AD_support=[]; MCI_nc_support=[]; MCI_c_support=[]; CN_support=[]

for i in range(len(AccDetails)):
    Details=AccDetails[i]

    A=Details['AD']['precision']
    AD_precision.append(A)
    A=Details['MCI_c']['precision']
    MCI_c_precision.append(A)    
    A=Details['MCI_nc']['precision']
    MCI_nc_precision.append(A)  
    A=Details['CN']['precision']
    CN_precision.append(A)  

    A=Details['AD']['recall']
    AD_recall.append(A)
    A=Details['MCI_c']['recall']
    MCI_c_recall.append(A)    
    A=Details['MCI_nc']['recall']
    MCI_nc_recall.append(A)  
    A=Details['CN']['recall']
    CN_recall.append(A)  
    
    A=Details['AD']['f1-score']
    AD_f1.append(A)
    A=Details['MCI_c']['f1-score']
    MCI_c_f1.append(A)    
    A=Details['MCI_nc']['f1-score']
    MCI_nc_f1.append(A)  
    A=Details['CN']['f1-score']
    CN_f1.append(A)  
    
    A=Details['AD']['support']
    AD_support.append(A)
    A=Details['MCI_c']['support']
    MCI_c_support.append(A)    
    A=Details['MCI_nc']['support']
    MCI_nc_support.append(A)  
    A=Details['CN']['support']
    CN_support.append(A)  

print('  ---------------------  ')
print('  ---------------------  ')
print('Mean of precision of AD :  ', np.mean(AD_precision))
print('Mean of precision of MCI_c :  ', np.mean(MCI_c_precision))
print('Mean of precision of MCI_nc :  ', np.mean(MCI_nc_precision))
print('Mean of precision of CN :  ', np.mean(CN_precision))
print('  ---------------------  ')
print('std of precision of AD :  ', np.std(AD_precision))
print('std of precision of MCI_c :  ', np.std(MCI_c_precision))
print('std of precision of MCI_nc :  ', np.std(MCI_nc_precision))
print('std of precision of CN :  ', np.std(CN_precision))

print('  ---------------------  ')
print('  ---------------------  ')
print('Mean of recall of AD :  ', np.mean(AD_recall))
print('Mean of recall of MCI_c :  ', np.mean(MCI_c_recall))
print('Mean of recall of MCI_nc :  ', np.mean(MCI_nc_recall))
print('Mean of recall of CN :  ', np.mean(CN_recall))
print('  ---------------------  ')
print('std of recall of AD :  ', np.std(AD_recall))
print('std of recall of MCI_c :  ', np.std(MCI_c_recall))
print('std of recall of MCI_nc :  ', np.std(MCI_nc_recall))
print('std of recall of CN :  ', np.std(CN_recall))

print('  ---------------------  ')
print('  ---------------------  ')
print('Mean of f1-score of AD :  ', np.mean(AD_f1))
print('Mean of f1-score of MCI_c :  ', np.mean(MCI_c_f1))
print('Mean of f1-score of MCI_nc :  ', np.mean(MCI_nc_f1))
print('Mean of f1-score of CN :  ', np.mean(CN_f1))
print('  ---------------------  ')
print('std of f1-score of AD :  ', np.std(AD_f1))
print('std of f1-score of MCI_c :  ', np.std(MCI_c_f1))
print('std of f1-score of MCI_nc :  ', np.std(MCI_nc_f1))
print('std of f1-score of CN :  ', np.std(CN_f1))

print('  ---------------------  ')
print('  ---------------------  ')
print('Mean of support of AD :  ', np.mean(AD_support))
print('Mean of support of MCI_c :  ', np.mean(MCI_c_support))
print('Mean of support of MCI_nc :  ', np.mean(MCI_nc_support))
print('Mean of support of CN :  ', np.mean(CN_support))
print('  ---------------------  ')
print('std of support of AD :  ', np.std(AD_support))
print('std of support of MCI_c :  ', np.std(MCI_c_support))
print('std of support of MCI_nc :  ', np.std(MCI_nc_support))
print('std of support of CN :  ', np.std(CN_support))
print('#########################################################################')
print('#########################################################################')


DataDict={"AccScores":AccScores,"AccDetails":AccDetails,"AllRegErrors":AllRegErrors
, "All_Predicts_class": All_Predicts_class , "All_Truth_class": All_Truth_class,
"All_Predicts_reg": All_Predicts_reg , "All_Truth_reg": All_Truth_reg}
import pickle
pickle.dump(DataDict,open("pkl_Results_Combined_2.pkl","wb"))




