import pdb
import sklearn
import os
import csv
import collections
import matplotlib.pyplot as plt
import shap
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
import numpy as np
from numpy import ndarray
import pandas as pd
from time import time as get_timestamp
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.calibration import CalibratedClassifierCV
from utils import COG_thresholding, ADD_thresholding
from config import root_path, checkpoint_dir_path, tb_log_path, late_embeddings_path, mid_embeddings_path, \
    early_embeddings_path, data_2_sq_path, mri_path

"""
Example Usage:
    model = NonImg_Model_Wrapper(
                            tasks=['COG', 'ADD'],                            # a list of tasks to train
                            main_config=read_json('main_config.json'),       # main_config is the dict read from json
                            task_config=read_json('task_config.json'),       # task config is the dict read from json
                            seed=1000)                                       # random seed
    model.train()                                                            # train the model
    thres = model.get_optimal_thres()                                        # get optimal threshold
    model.gen_score(['test'], thres)                                         # generate csv files to future evaluation

for more details how this class is called, please see main.py

note: in the tasks argument, need to put COG before ADD since imputer will be calculated based on COG data
      and the imputer will be used to transform the ADD data 
"""

class NonImg_Model_Wrapper:
    def __init__(self, tasks, main_config, task_config, seed):
        # --------------------------------------------------------------------------------------------------------------
        # some constants
        self.seed = seed  # random seed number
        self.model_name = main_config['model_name']  # user assigned model_name, will create folder using model_name to log
        self.csv_dir = main_config['csv_dir']  # data will be loaded from the csv files specified in this directory
        self.config = task_config  # task_config contains task specific info
        self.n_tasks = len(tasks)  # number of tasks will be trained
        self.tasks = tasks  # a list of tasks names to be trained
        self.features = task_config['features'] # a list of features
        if 'ADD_score' in self.features:
            self.features.remove('ADD_score')
        if 'COG_score' in self.features:
            self.features.remove('COG_score')

        # --------------------------------------------------------------------------------------------------------------
        # folders preparation to save checkpoints of model weights *.pth
        self.checkpoint_dir = os.path.abspath(os.path.join(checkpoint_dir_path, self.model_name))
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        # folders preparation to save tensorboard and other logs
        self.tb_log_dir = os.path.abspath(os.path.join(tb_log_path, self.model_name))
        if not os.path.exists(self.tb_log_dir):
            os.mkdir(self.tb_log_dir)

        # --------------------------------------------------------------------------------------------------------------
        # initialize models
        self.models = []  # note: self.models[i] is for the i th task
        self.init_models([task_config[t]['name'] for t in tasks])

        # --------------------------------------------------------------------------------------------------------------
        # initialize data
        self.train_data = []                      # note: self.train_data[i] contains the
        self.imputer = None
        self.train_set = pd.read_csv(os.path.join(self.csv_dir, 'train.csv'))
        self.test_set = pd.read_csv(os.path.join(self.csv_dir, 'test.csv'))
        self.train_set = self.train_set.drop(['RID', 'VISCODE', 'filename', 'benefit'], axis=1)
        self.test_set = self.test_set.drop(['RID', 'VISCODE', 'filename', 'benefit'], axis=1)
        """
        self.train_set = pd.read_csv(os.path.join(self.csv_dir, 'preprocessed_train.csv'))
        self.valid_set = pd.read_csv(os.path.join(self.csv_dir, 'preprocessed_valid.csv'))
        self.load_preprocess_data()               #       features and labels for the i th task
        self.valid_set = self.load_dataset(os.path.join(self.csv_dir, 'valid.csv'))
        self.test_set = self.load_dataset(os.path.join(self.csv_dir, 'test.csv'))
        self.train_data[0].to_csv(os.path.join(self.csv_dir, 'preprocessed_train.csv'), index=False)
        self.valid_set.to_csv(os.path.join(self.csv_dir, 'preprocessed_valid.csv'), index=False)
        self.test_set.to_csv(os.path.join(self.csv_dir, 'preprocessed_test.csv'), index=False)
        """

        print('初始化完毕')

    def train(self):
        x_train, y_train = self.train_set.drop(['COG'], axis=1).to_numpy(), self.train_set['COG'].values
        x_test, y_test = self.test_set.drop(['COG'], axis=1).to_numpy(), self.test_set['COG'].values
        model = CatBoostRegressor(iterations=1, learning_rate=0.05)
        for epoch in range(100):
            # 训练、保存模型
            if epoch != 0:
                init_model = model
            else:
                init_model = None
            model.fit(x_train, y_train, init_model=init_model, verbose=False)
            model.save_model(os.path.join(self.checkpoint_dir, 'CatBoostRegressor_{}'.format(epoch)))

            # 验证模型
            pred_train = self.get_three_classes_prediction(model.predict(x_train))
            train_accuracy = sum(pred_train == y_train) / y_train.shape[0]
            pred_test = self.get_three_classes_prediction(model.predict(x_test))
            test_accuracy = sum(pred_test == y_test) / y_test.shape[0]
            print('Epoch {}: train_accuracy={:.4f} -- test_accuracy={:.4f}'.format(
                epoch + 1, train_accuracy, test_accuracy
            ))

    def get_three_classes_prediction(self, scores: ndarray, thresholds: tuple = (0.5, 1.5)) -> ndarray:
        prediction = np.zeros(len(scores), 'int')
        for i in range(len(scores)):
            if scores[i] > thresholds[1]:
                prediction[i] = 2
            elif scores[i] >= thresholds[0]:
                prediction[i] = 1
        return prediction

    def get_optimal_thres(self, csv_name='valid'):
        self.gen_score(stages=[csv_name])
        thres = {}
        for i, task in enumerate(self.tasks):
            if task == 'COG' and self.config['COG']['type'] == 'reg':
                thres['NC'], thres['DE'] = COG_thresholding(self.tb_log_dir + csv_name + '_eval.csv')
            elif task == 'ADD':
                thres[task] = ADD_thresholding(self.tb_log_dir + csv_name + '_eval.csv')
            else:
                print("optimal for the task {} is not supported yet".format(task))
        return thres

    def gen_score(self, stages=['train', 'valid', 'test', 'OASIS'], thres={'ADD':0.5, 'NC':0.5, 'DE':1.5}):
        for stage in stages:
            data = pd.read_csv(self.csv_dir + stage + '.csv')[self.features + self.tasks + ['filename']]
            data = self.drop_cases_without_label(data, 'COG')
            COG_data = self.preprocess_pipeline(data[self.features+['COG']], 'COG') # treat it as COG data to do the preprocessing
            features = COG_data.drop(['COG'], axis=1)
            labels = data[self.tasks]
            filenames = data['filename']

            # make sure the features and labels has the same number of rows
            if len(features.index) != len(labels.index):
                raise ValueError('number of rows between features and labels have to be the same')

            predicts = []
            for i, task in enumerate(self.tasks):
                if task == 'COG':
                    predicts.append(self.models[i].predict(features))
                    print("the shape of prediction for COG task is ", predicts[-1].shape)
                if task == 'ADD':
                    predicts.append(self.models[i].predict_proba(features))
                    print("the shape of prediction for ADD task is ", predicts[-1].shape)

            content = []
            for i in range(len(features.index)):
                label = labels.iloc[i] # the feature and label are for the i th subject
                filename = filenames.iloc[i]
                case = {'filename': filename}
                for j, task in enumerate(self.tasks): # j is the task index
                    case[task] = "" if np.isnan(label[task]) else int(label[task])
                    if task == 'COG':
                        case[task+'_score'] = predicts[j][i]
                        if case[task+'_score'] < thres['NC']:
                            case[task + '_pred'] = 0
                        elif thres['NC'] <= case[task+'_score'] <= thres['DE']:
                            case[task + '_pred'] = 1
                        else:
                            case[task + '_pred'] = 2
                    elif task == 'ADD':
                        case[task + '_score'] = predicts[j][i, 1]
                        if case[task+'_score'] < thres['ADD']:
                            case[task + '_pred'] = 0
                        else:
                            case[task + '_pred'] = 1
                content.append(case)

            with open(self.tb_log_dir + stage + '_eval.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(content[0].keys()))
                writer.writeheader()
                for case in content:
                    writer.writerow(case)

    def shap(self, stage='test'):
        """
        This function will generate shap value for a specific stage
        if stage is 'test', the shap analysis will be performed on the testing section of the data
        """
        # get the data ready
        data = pd.read_csv(self.csv_dir + stage + '.csv')
        task_data = []
        for task in self.tasks:
            task_data.append(data[self.features + [task]])
        for i, task in enumerate(self.tasks):
            task_data[i] = self.preprocess_pipeline(task_data[i], task).drop([task], axis=1)

        # get the explainer ready
        self.explainer = []
        shap_values = []
        task_names = [self.config[t]['name'] for t in self.tasks]
        for i, task in enumerate(self.tasks):
            # background = shap.maskers.Independent(self.train_data[i], max_samples=100) # can we sample background from train_data?
            if task_names[i] in ['XGBoostCla', 'XGBoostReg']:
                if 'Reg' in task_names[i]:
                    explainer = shap.TreeExplainer(self.models[i], task_data[i], model_output='raw')
                    shap_values.append(explainer.shap_values(task_data[i]))
                elif 'Cla' in task_names[i]:
                    explainer = shap.TreeExplainer(self.models[i], task_data[i], model_output='predict_proba')
                    shap_values.append(explainer.shap_values(task_data[i])[1]) # index 1 means only taking ADD prob
            elif task_names[i] in ['CatBoostCla', 'CatBoostReg']:
                if 'Reg' in task_names[i]:
                    explainer = shap.TreeExplainer(self.models[i], model_output='raw')
                    shap_values.append(explainer.shap_values(task_data[i]))
                elif 'Cla' in task_names[i]: # use kernel explainer becaus shap only support model_output="raw"
                    explainer = shap.KernelExplainer(self.models[i].predict_proba, task_data[i])
                    shap_values.append(explainer.shap_values(task_data[i], nsamples=200)[1])
            elif task_names[i] in ['RandomForestCla', 'RandomForestReg', 'DecisionTreeCla', 'DecisionTreeReg']:
                if 'Reg' in task_names[i]:
                    explainer = shap.TreeExplainer(self.models[i], task_data[i], model_output='raw')
                    shap_values.append(explainer.shap_values(task_data[i]))
                elif 'Cla' in task_names[i]:
                    explainer = shap.TreeExplainer(self.models[i], task_data[i], model_output='probability')
                    shap_values.append(explainer.shap_values(task_data[i])[1])
            elif task_names[i] in ['PerceptronCla', 'PerceptronReg']:
                if 'Reg' in task_names[i]:
                    explainer = shap.KernelExplainer(self.models[i].predict, task_data[i])
                    shap_values.append(explainer.shap_values(task_data[i], nsamples=500))
                elif 'Cla' in task_names[i]:
                    explainer = shap.KernelExplainer(self.models[i].predict_proba, task_data[i])
                    shap_values.append(explainer.shap_values(task_data[i], nsamples=500)[1]) # index 1 means only taking ADD prob
            elif task_names[i] in ['SupportVectorCla', 'SupportVectorReg', 'NearestNeighborCla', 'NearestNeighborReg']:
                if 'Reg' in task_names[i]:
                    explainer = shap.KernelExplainer(self.models[i].predict, task_data[i])
                    shap_values.append(explainer.shap_values(task_data[i], nsamples=200))
                elif 'Cla' in task_names[i]:
                    explainer = shap.KernelExplainer(self.models[i].predict_proba, task_data[i])
                    shap_values.append(explainer.shap_values(task_data[i], nsamples=200)[1]) # index 1 means only taking ADD prob
            print(task + "'s shap values in shape: ", shap_values[-1].shape)
            # save the shap_values into a csv file for future use
            # rows are subjects, columns are the features
            columns = task_data[i].columns
            df = pd.DataFrame(shap_values[-1], columns=columns)
            df.to_csv(self.tb_log_dir + 'shap_'+stage+'_'+task+'.csv', index = False, header=True)
            self.shap_beeswarm_plot(shap_values[-1], task_data[i], task, stage)
        return shap_values, task_data


    ###############################################################################################################
    # below methods are internal methods and won't be called from outside of the class
    def init_models(self, task_models):
        """
        each task can have different types of models
        for example, we will use regression relevant model for the COG task
                     and classification relevant model for the ADD task
        the task_models parameter should be a python list
                     where the task_models[i] is the name of the model for the i th task
        after model initialization, models will be appended into self.models
                     where the self.models[i] is for the i th task
        """
        for name in task_models:
            # random forest model tested, function well
            if name == 'RandomForestCla':
                model = RandomForestClassifier()
            elif name == 'RandomForestReg':
                model = RandomForestRegressor()

            # xgboost model tested, function well
            elif name == 'XGBoostCla':
                model = xgb.XGBClassifier(use_label_encoder=False)
            elif name == 'XGBoostReg':
                model = xgb.XGBRegressor()

            # catboost model tested, function well
            elif name == 'CatBoostCla':
                model = CatBoostClassifier()
            elif name == 'CatBoostReg':
                model = CatBoostRegressor()

            # mlp model tested, function well
            elif name == 'PerceptronCla':
                model = MLPClassifier(max_iter=1000)
            elif name == 'PerceptronReg':
                model = MLPRegressor(max_iter=1000)

            # decision tree model tested, function well
            elif name == 'DecisionTreeCla':
                model = DecisionTreeClassifier()
            elif name == 'DecisionTreeReg':
                model = DecisionTreeRegressor()

            # support vector model tested, function well
            elif name == 'SupportVectorCla':
                model = SVC(probability=True)
            elif name == 'SupportVectorReg':
                model = SVR()

            # KNN model tested, function well
            elif name == 'NearestNeighborCla':
                model = KNeighborsClassifier()
            elif name == 'NearestNeighborReg':
                model = KNeighborsRegressor()

            self.models.append(model)

    def init_imputer(self, data):
        """
        since cases with ADD labels is only a subset of the cases with COG label
        in this function, we will initialize a single imputer
        and fit the imputer based on the COG cases from the training part
        """
        imputation_method = self.config['impute_method']
        #print(imputation_method)
        if imputation_method == 'mean':  # 均值填充
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        elif imputation_method == 'median':  # 中位数填充
            imp = SimpleImputer(missing_values=np.nan, strategy='median')
        elif imputation_method == 'most_frequent':  # 众数填充
            imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        elif imputation_method == 'constant':  # 常数0填充
            imp = SimpleImputer(missing_values=np.nan, strategy='constant')
        elif imputation_method == 'KNN':  # K邻近法填充
            imp = KNNImputer(n_neighbors=20)
        elif imputation_method == 'Multivariate':  # 多元填补
            imp = IterativeImputer(max_iter=1000)
        else:
            raise NameError('method for imputation not supported')
        imp.fit(data)
        return imp

    def load_preprocess_data(self):
        data_train = pd.read_csv(os.path.realpath(os.path.join(self.csv_dir, 'train.csv')))
        #print("data_train.columns= ", list(data_train.columns))
        #print("data_train.len= ", len(data_train))
        pdb.set_trace()
        for task in self.tasks:
            #print("task= ", task)
            #print("self.features + [task]= ", self.features)
            columns = list(set(self.features + [task]) & set(data_train.columns))
            self.train_data.append(data_train[columns])
            #print("==========")
        for i, task in enumerate(self.tasks):
            self.train_data[i] = self.preprocess_pipeline(self.train_data[i], task)
            # print('after preprocess pipeline, the data frame for the {} task is'.format(task))
            # print(self.train_data[i])
            # print('\n' * 2)

    def load_dataset(self, dataset_path):
        dataset = pd.read_csv(dataset_path)
        columns = list(set(self.features + ['COG']) & set(dataset.columns))
        return self.preprocess_pipeline(dataset[columns], 'COG')

    def preprocess_pipeline(self, data, task):
        """
        Cathy, we need to remove cases with too much missing non-imaging features, please consider adding the step
        """
        # data contains features + task columns
        data = self.drop_cases_without_label(data, task)
        data = self.transform_categorical_variables(data)
        features = data.drop([task], axis=1) # drop the task columns to get all features
        #print("process_pipeline   features= ", features)
        #print("len(features)= ", len(features))    #len= 66
        features = self.imputation(features) # do imputation merely on features
        # features.to_csv(os.path.join(root_path, 'demo_train.csv'), index=False)
        features = self.normalize(features)  # normalize features
        features[task] = data[task]          # adding the task column back
        return features                      # return the complete data

    def drop_cases_without_label(self, data, label):
        data = data.dropna(axis=0, how='any', subset=[label], inplace=False)
        return data.reset_index(drop=True)

    def transform_categorical_variables(self, data):
        if 'gender' in data:
            return data.replace({'male': 0, 'female': 1})
        else:
            return data
        # return pd.get_dummies(data, columns=['gender'])

    def imputation(self, data):
        columns = data.columns
        #print("len(columns)= ", len(columns))
        #print("columns= ", columns)
        if self.imputer == None:
            self.imputer = self.init_imputer(data)
            #print("==comming !!!==")
            #print("self.imputer.len= ", np.shape(self.imputer))
        #print("data.shape= ", np.shape(data))
        #print("type(data)= ", type(data))
        # arr = np.zeros((789, 66))
        # data11 = pd.DataFrame(arr, columns=columns)
        # print("data11.shape= ", np.shape(data11))
        data = self.imputer.transform(data)
        #print("after data.shape= ", np.shape(data))
        return pd.DataFrame(data, columns=columns)

    def normalize(self, data):
        df_std = data.copy()
        for column in df_std.columns:
            if data[column].std(): # normalize only when std != 0
                df_std[column] = (data[column] - data[column].mean()) / data[column].std()
        return df_std

    def shap_beeswarm_plot(self, shap_values, data, task, stage):
        from matplotlib import rc, rcParams
        rc('axes', linewidth=2)
        rc('font', weight='bold')
        fig, ax = plt.subplots(figsize=(8, 10))
        fig.text(-0.04, 0.87, 'Features', fontsize=15, fontweight='black')
        shap.summary_plot(shap_values, data)
        ax.set_xlabel('SHAP value', fontsize=15, fontweight='black')
        plt.savefig(self.tb_log_dir + '{}_shap_beeswarm_{}.png'.format(stage, task), dpi=100, bbox_inches='tight')
        plt.close()


class Fusion_Model_Wrapper(NonImg_Model_Wrapper):

    def __init__(self, *args, **kwargs):
        super(Fusion_Model_Wrapper, self).__init__(*args, **kwargs)
        self.train_set = pd.read_csv(os.path.join(self.csv_dir, 'fusion_train.csv'))
        self.valid_set = pd.read_csv(os.path.join(self.csv_dir, 'fusion_valid.csv'))

        """
        # 加入COG_score作为输入
        from model_eval.mri_predict_test_set import MRIModel
        cp_dict = {}
        checkpoint_dir_path = os.path.join(root_path, 'checkpoint_dir/MRI_only_v3')
        for fn in os.listdir(checkpoint_dir_path):
            cp_type, index = fn[:fn.rfind('.')].split('_')
            index = int(index)
            if index in cp_dict:
                cp_dict[index][cp_type] = os.path.join(checkpoint_dir_path, fn)
            else:
                cp_dict[index] = {cp_type: os.path.join(checkpoint_dir_path, fn)}
        mri_test_scores = np.load(os.path.join(root_path, 'model_eval/eval_result/mri/scores.npy'))
        mri_eval_result = pd.read_csv(os.path.join(root_path, 'model_eval/eval_result/mri/result.csv'))
        df = pd.DataFrame({
            'model_index': sorted(cp_dict.keys()),
            'accuracy': mri_eval_result['accuracy'].values
        }).sort_values('accuracy', ascending=False)
        best_cp = cp_dict[df.iloc[0, 0]]
        best_model = MRIModel(best_cp['backbone'], best_cp['COG'])

        def mri_generator(filenames):
            for filename in filenames:
                mri = np.load(os.path.join(mri_path, filename))
                mri = np.expand_dims(np.expand_dims(mri, axis=0), axis=0)
                yield mri

        source_train_set = pd.read_csv(os.path.join(root_path, 'lookupcsv/CrossValid/no_cross/train_source.csv'))
        train_cog_score = best_model.predict(
            mri_generator(source_train_set['filename'].values),
            source_train_set.shape[0]
        )
        source_valid_set = pd.read_csv(os.path.join(root_path, 'lookupcsv/CrossValid/no_cross/valid_source.csv'))
        valid_cog_score = best_model.predict(
            mri_generator(source_valid_set['filename'].values),
            source_valid_set.shape[0]
        )
        test_cog_score = mri_test_scores[df.index[0]]
        test_set = pd.read_csv(os.path.join(self.csv_dir, 'preprocessed_test.csv'))
        self.train_set.insert(self.train_set.shape[1], 'COG_score', train_cog_score, allow_duplicates=False)
        self.valid_set.insert(self.valid_set.shape[1], 'COG_score', valid_cog_score, allow_duplicates=False)
        test_set.insert(test_set.shape[1], 'COG_score', test_cog_score, allow_duplicates=False)
        self.train_set.to_csv(os.path.join(self.csv_dir, 'fusion_train.csv'), index=False)
        self.valid_set.to_csv(os.path.join(self.csv_dir, 'fusion_valid.csv'), index=False)
        test_set.to_csv(os.path.join(self.csv_dir, 'fusion_test.csv'), index=False)
        """

