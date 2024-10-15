import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    accuracy_score
)
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score
import lightgbm as lgb
import xgboost as xgb
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import matplotlib.pyplot as plt
import category_encoders as ce
pd.set_option('display.max_columns', None)


# Model Visualization script
class ModelViz:

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_optimum_threshold(df, target='Target', score='Score'):
        '''
        Given probability scores and binary target, returns the optimum cut off point, where 
        `true positive rate` is high and `false positive rate` is low.

        Parameters
        ----------
        df: pandas.DataFrame, Dataframe that contains binary target values - 0 & 1 and prediction scores
        target: str, Name of the target column. Default = Target
        score: str, Name of the probability score column. Default = Score

        Returns
        -------
        float: returns ROC AUC
        pd.DataFrame: returns a new DataFrame that provides tpr, fpr and optimum threshold
        matplotlib.pyplot: returns a ROC curve with cut-off point
        matplotlib.pyplot: returns a Target Separability Plot with threshold

        '''
        fpr, tpr, thresholds = metrics.roc_curve(df[target], df['Score'])
        roc_auc = metrics.auc(fpr, tpr)

        ####################################
        # The optimal cut off would be where tpr is high and fpr is low
        # tpr - (1-fpr) is zero or near to zero is the optimal cut off point
        ####################################
        i = np.arange(len(tpr))  # index for df
        roc = pd.DataFrame({
            'fpr': pd.Series(fpr, index=i),
            'tpr': pd.Series(tpr, index=i),
            '1-fpr': pd.Series(1-fpr, index=i),
            'tf': pd.Series(tpr - (1-fpr), index=i),
            'thresholds': pd.Series(thresholds, index=i)
        })

        cutoff_df = roc.iloc[(roc.tf-0).abs().argsort()
                             [:1]].reset_index(drop=True)

        # # Plot tpr vs 1-fpr
        # fig, ax = plt.subplots()
        # plt.plot(roc['tpr'], label='tpr')
        # plt.plot(roc['1-fpr'], color='red', label='1-fpr')
        # plt.xlabel('1-False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic')
        # ax.set_xticklabels([])
        # plt.legend()

        # # Plot tpr vs 1-fpr
        # fig2, ax2 = plt.subplots()
        # sns.kdeplot(x=df[df[target] == 0]['Score'], label='0')
        # sns.kdeplot(x=df[df[target] == 1]['Score'], label='1')
        # plt.axvline(x=cutoff_df['thresholds'].values[0],
        #             label='thresh={:.2f}'.format(cutoff_df['thresholds'].values[0]), color='red', ls='--')
        # plt.title('Target Separability')
        # plt.legend()

        return roc_auc, cutoff_df # , ax, ax2


    @staticmethod
    def get_classification_report(clf, X, y, thres=0.5):
        '''
        Given model, X and y provides classification report for the model

        Parameters
        ----------
        clf: sklearn model, trained classification model that has predict_proba available
        X: pandas.DataFrame or numpy array, Dataframe/array that acts as independent variables for the model
        y: pandas.Series or numpy 1D-array, Series/1D-array that acts as the dependant/target variable for the model
        thres: float, optional. The probability threshold to determine 0 or 1. Default is 0.5

        Returns
        -------
        classification report: str, returns classification report

        '''

        x_train_proba = clf.predict_proba(X)[:, 1]
        x_train_pred = np.where(x_train_proba > thres, 1, 0)

        clf_report = metrics.classification_report(y, x_train_pred)

        return clf_report

class DataGenerator:
    def __init__(self, master_file_path, n_samples):
        
        master_file = pd.read_excel(master_file_path, skiprows = 1)
        # Id column
        self.id = 'cust_id'
        # Continuous cols
        self.cont_col = master_file[master_file['Variable Type'].isin(['Continuous', 'Integer'])]['Analytics_Nomenclature'].tolist()
        self.cont_col.remove('pincode')
        # Categorical cols
        self.cat_col = master_file[master_file['Variable Type'].isin(['Categorical'])]['Analytics_Nomenclature'].tolist()

        # Binary cols
        self.bin_col = master_file[master_file['Variable Type'].isin(['Binary'])]['Analytics_Nomenclature'].tolist()

        # 
        self.date_col = master_file[master_file['Variable Type'].isin(['Datetime'])]['Analytics_Nomenclature'].tolist()[0]
        np.random.seed(42)
        self.n_samples = n_samples
        self.continuous_data = None
        self.categorical_data = None
        self.final_data = None

    def generate_categorical_data(self):
        cat_variables = {
            'account_type': ['SAVINGS', 'CURRENT'],
            'cust_segment': ['Gold', 'Silver', 'Platinum'],
            'education': ['Below 10th', 'UG', 'Graduate', 'PG'],
            'gender': ['MALE', 'FEMALE'],
            'marital_status': ['Married', 'Single/Unmarried', 'Divorced/Separated'],
            'occupation': ['SELF EMPLOYED', 'SALARIED', 'BUSINESS OWNER', 'NON-EARNER'],
            'state': ['Maharashtra', 'Gujarat', 'West Bengal', 'Uttar Pradesh', 'Tamil Nadu', 'Haryana', 'Rajasthan', 'Punjab',
                        'Delhi', 'Madhya Pradesh', 'Out of India', 'Andhra Pradesh', 'Karnataka', 'Telangana', 'Odisha', 'Kerala',
                        'Bihar', 'Assam', 'Chhattisgarh', 'Jharkhand', 'Uttarakhand', 'Himachal Pradesh', 'Jammu and Kashmir',
                        'Chandigarh', 'Goa', 'Manipur', 'Nagaland', 'Tripura', 'Puducherry', 'Dadra and Nagar Haveli and Daman and Diu',
                        'Sikkim', 'Arunachal Pradesh', 'Meghalaya', 'Mizoram', 'Andaman and Nicobar'],
            'city': ['Mumbai', 'Bangalore', 'Hyderabad', 'Pune', 'Kolkata', 'Delhi', 'Chennai', 'Others'],
            'zone': ['North', 'East', 'West', 'South'],
            'current_city': ['Mumbai', 'Bangalore', 'Hyderabad', 'Pune', 'Kolkata', 'Delhi', 'Chennai', 'Others']
        }

        # Probabilities for categorical variables
        prob = {
            'account_type': [0.80, 0.20],
            'cust_segment': [0.5, 0.25, 0.25],
            'education': [0.25, 0.25, 0.25, 0.25],
            'gender': [0.6, 0.4],
            'marital_status': [0.6, 0.35, 0.05],
            'occupation': [0.404, 0.339, 0.152, 0.105],
            'state': [0.119496018, 0.089888389, 0.07827775, 0.072670547, 0.064865747, 0.062475027, 0.050854399, 0.049988679,
                        0.046685624, 0.043482459, 0.036553368, 0.033316907, 0.03284409, 0.031895128, 0.030473349, 0.029424496,
                        0.028928372, 0.020664074, 0.018809435, 0.015855997, 0.007232093, 0.006562825, 0.005510642, 0.005430729,
                        0.005330838, 0.00238739, 0.001664846, 0.001534988, 0.001481713, 0.001328547, 0.001288591, 0.001175382,
                        0.00107882, 0.000526091, 1.66485E-05],
            'city': [0.4, 0.1, 0.02, 0.01, 0.2, 0.1, 0.1, 0.07],
            'zone': [0.25, 0.25, 0.25, 0.25],
            'current_city': [0.4, 0.1, 0.02, 0.01, 0.2, 0.1, 0.1, 0.07]
        }

        # List of flag variables
        flag_variables = self.bin_col
        for flag_var in flag_variables:
            cat_variables[flag_var] = ['Y', 'N']
            prob[flag_var] = [0.5, 0.5]

        # Categorical data
        self.categorical_data = pd.DataFrame({col: np.random.choice(cat_variables[col], self.n_samples, p=prob[col]) for col in cat_variables})
        self.categorical_data[self.id] = self.categorical_data.index
        return self.categorical_data

    def generate_continuous_data(self):
        self.continuous_data = pd.DataFrame()
        for col in self.cont_col:
            self.continuous_data[col] = np.random.randint(1, 100, size=self.n_samples)

        # Generate KYC_LAST_DONE_DATE between 01-01-2020 and 01-12-2023
        start_date = np.datetime64('2020-01-01')
        end_date = np.datetime64('2023-12-01')
        self.continuous_data[self.date_col] = np.random.choice(pd.date_range(start_date, end_date), size=self.n_samples)
        return self.continuous_data

    def combine_data(self):
        # Combine categorical and continuous data
        self.final_data = pd.concat([self.categorical_data, self.continuous_data], axis=1)

    def generate_dataset(self):
        self.generate_continuous_data()
        self.generate_categorical_data()
        self.combine_data()
        return self.final_data

class data_preprocess():

    def __init__(self, master_file_path, input_data_path, target, test_split, generate_random_data=None, n_samples = None, validation=None):
        
        self.master_file_path = master_file_path
        master_file = pd.read_excel(self.master_file_path, skiprows = 1)

        self.n_samples = n_samples
        # Id column
        self.id = 'cust_id'

        # Continuous cols
        self.cont_col = master_file[master_file['Variable Type'].isin(['Continuous', 'Integer'])]['Analytics_Nomenclature'].tolist()
        self.cont_col.remove('pincode')
        
        # Categorical cols
        self.cat_col = master_file[master_file['Variable Type'].isin(['Categorical'])]['Analytics_Nomenclature'].tolist()

        # Binary cols
        self.bin_col = master_file[master_file['Variable Type'].isin(['Binary'])]['Analytics_Nomenclature'].tolist()

        # Date cols
        self.date_col = master_file[master_file['Variable Type'].isin(['Datetime'])]['Analytics_Nomenclature'].tolist()[0]

        # dataframe
        if generate_random_data is None:
            self.df = pd.read_csv(input_data_path)
        else:
            data_generator = DataGenerator(master_file_path=self.master_file_path, n_samples=self.n_samples)
            self.df = data_generator.generate_dataset()

        # target 
        self.target = target

        #train
        self.train = None

        #test
        self.test = None

        #valid
        self.valid = None

        # Validation to be split from train test
        self.validation = validation

        # Test Split
        self.test_split = test_split

        #Features
        self.s_features = None

    # Chi-Square Test and Cramer's v
    @staticmethod
    def chi_sq_test(df, x,y):
        cross_tabs = pd.crosstab(df[x], df[y])
        chi2, p, dof, con_table = stats.chi2_contingency(cross_tabs)
        if p < 0.05:
            decision = 'Reject H0: there is significant association between ' + x + ' and ' + y
            # calculating cramer's v
            n = cross_tabs.sum().sum()
            minimum_dimension = min(cross_tabs.shape)-1
            v = np.sqrt(chi2/(n*dof))
            if v <= 0.2:
                strength = 'Weak Association between '  + x + ' and ' + y
            elif v > 0.2 and v <= 0.6:
                strength = 'Medium Association between '  + x + ' and ' + y
            else:
                strength = 'Strong Association between '  + x + ' and ' + y
        else: 
            decision = 'Do not reject H0: There is no relation between ' + x + ' and ' + y
            strength = 'No association between '  + x + ' and ' + y
            v = 0
        # print(f'chi-squared = {chi2}\np value= {p}\ndegrees of freedom = {dof}')
        # print(decision)
        # print("Cramer's V: " + str(v))
        # print(strength)
        return p

    @staticmethod
    def cont_test(df, x, target):
        # Perform the two sample t-test with equal variances
        t = stats.ttest_ind(a=df[df[target]==1][x], b=df[df[target]==0][x], equal_var=True)
        # print(df[df[target]==1][x].mean(), df[df[target]==0][x].mean())
        return t.pvalue

    @staticmethod
    # Function to create buckets
    def assign_bucket(var, ranges, label):
        for i, range in enumerate(ranges):
            if var > range:
                return label[i]
        return label[-1] if var > range[-1] else 'Not Available'

    def data_prep(self):
        # Auto-Correction of categorical variables
        for cat in self.cat_col:
            self.df[cat] = self.df[cat].str.strip().str.upper()
            if self.df[cat].nunique() > 4:
                temp_cat = self.df[cat].value_counts().reset_index().sort_values('count', ascending=False)
                self.df[cat] = np.where(self.df[cat].isin(temp_cat[cat][0:3]), self.df[cat], 'OTHERS')
        
        self.bin_col.remove(self.target)

        for bin in self.bin_col:
            self.df[bin] = self.df[bin].str.strip().str.upper()
            self.df[bin] = np.where(self.df[bin]=='Y', 'Y', 'N')
        
        # Treating the date column
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df = self.df[~self.df[self.date_col].isna()].reset_index(drop = True)
        self.df['KYC_RECENCY_YEARS'] = (datetime.now() - self.df[self.date_col])/np.timedelta64(1, 'D')/365
        kyc_ranges = [0.5, 1, 3, 5]
        kyc_labels = ['Less than 6 Months', '6 Months to 1 Year', '1 to 3 Years', '3 to 5 Years', 'More than 5 Years']
        self.df['KYC_RECENCY_BUCKET'] = self.df['KYC_RECENCY_YEARS'].apply(self.assign_bucket, args = (kyc_ranges,kyc_labels))
        self.cat_col.append('KYC_RECENCY_BUCKET')
        # Converting target into binary
        self.df[self.target] = np.where(self.df[self.target]=='Y', 1, 0)

    ## EDA part to be integrated

    def data_split(self):
        if self.validation is None:
            self.train, self.test = train_test_split(self.df, test_size=self.test_split, random_state=42, stratify=self.df[self.target])
        else:
            self.train, test_valid = train_test_split(self.df, test_size=self.test_split, random_state=42, stratify=self.df[self.target])
            self.test, self.valid = train_test_split(test_valid, test_size=0.5, random_state=42, stratify=test_valid[self.target])   

    def data_transformation(self):
        if self.validation is None:
            imputer = KNNImputer(n_neighbors=5)
            imputer_fit = imputer.fit(self.train[self.cont_col])
            self.train[self.cont_col] = imputer_fit.fit_transform(self.train[self.cont_col])
            self.test[self.cont_col] = imputer_fit.transform(self.test[self.cont_col])

            scaler = StandardScaler()
            scaler_fit = scaler.fit(self.train[self.cont_col])

            self.train[self.cont_col] = scaler_fit.fit_transform(self.train[self.cont_col])
            self.test[self.cont_col] = scaler_fit.transform(self.test[self.cont_col])

            # Target based encoding on training
            category_cols = self.cat_col + self.bin_col
            TE = ce.TargetEncoder()
            TE_fit = TE.fit(X=self.train[category_cols], y=self.train[self.target])

            self.train[category_cols] = TE_fit._transform(X = self.train[category_cols])
            self.test[category_cols] = TE_fit._transform(X = self.test[category_cols])
        else:
            imputer = KNNImputer(n_neighbors=5)
            imputer_fit = imputer.fit(self.train[self.cont_col])
            self.train[self.cont_col] = imputer_fit.fit_transform(self.train[self.cont_col])
            self.test[self.cont_col] = imputer_fit.transform(self.test[self.cont_col])
            self.valid[self.cont_col] = imputer_fit.transform(self.valid[self.cont_col])

            scaler = StandardScaler()
            scaler_fit = scaler.fit(self.train[self.cont_col])

            self.train[self.cont_col] = scaler_fit.fit_transform(self.train[self.cont_col])
            self.test[self.cont_col] = scaler_fit.transform(self.test[self.cont_col])
            self.valid[self.cont_col] = scaler_fit.transform(self.valid[self.cont_col])

             # Target based encoding on training
            category_cols = self.cat_col + self.bin_col
            TE = ce.TargetEncoder()
            TE_fit = TE.fit(X=self.train[category_cols], y=self.train[self.target])

            self.train[category_cols] = TE_fit._transform(X = self.train[category_cols])
            self.test[category_cols] = TE_fit._transform(X = self.test[category_cols])
            self.valid[category_cols] = TE_fit._transform(X = self.valid[category_cols])

    def variable_selection(self):
        # cont Features selections
        Row = 0
        for i in self.cont_col:
            out = self.cont_test(self.train, i, self.target)
            if out < 0.05:
                Row = Row + 1
                if Row == 1:
                    significant = [i]
                else:
                    significant = significant + [i]


        # cat Features selections
        Row = 0
        for i in self.cat_col + self.bin_col:
            out = self.chi_sq_test(self.train, i, self.target)
            if out < 0.05:
                Row = Row + 1
                if Row == 1:
                    significant_cat = [i]
                else:
                    significant_cat = significant_cat + [i]

        self.s_features = significant_cat + significant
        print('Final features selected for model development: \n', self.s_features)

    def run_pre_processing(self):
        self.data_prep()
        self.data_split()
        self.variable_selection()
        self.data_transformation()
        if self.validation is None:
            return self.train, self.test, self.s_features
        else:
            return self.train, self.test, self.valid, self.s_features

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK

class model_evaluate:

    def __init__(self, train, eval, target, s_feature, validation = None):
        self.train = train
        self.eval = eval
        self.validation = validation
        self.target = target
        self.s_feature = s_feature

    # Decile summary generation function
    @staticmethod
    def decile_fun(score, prob_list):
        if score >= prob_list[0]:
            decile = 1
        elif score >= prob_list[1] < prob_list[0]:
            decile = 2
        elif score >= prob_list[2] < prob_list[1]:
            decile = 3
        elif score >= prob_list[3] < prob_list[2]:
            decile = 4
        elif score >= prob_list[4] < prob_list[3]:
            decile = 5
        elif score >= prob_list[5] < prob_list[4]:
            decile = 6
        elif score >= prob_list[6] < prob_list[5]:
            decile = 7
        elif score >= prob_list[7] < prob_list[6]:
            decile = 8
        elif score >= prob_list[8] < prob_list[7]:
            decile = 9
        else:
            decile = 10
        return decile

    # Decile Summary
    def decile_summary(self, prob, actual, prob_list, req_dig=True):
        Decile = [self.decile_fun(col, prob_list) for col in prob]
        results = pd.DataFrame({'Sum': actual,'Count': actual,'Probability': prob}).reset_index(drop = True)
        results['Decile'] = Decile
        decile_sum = results.groupby('Decile')['Sum'].sum().reset_index()
        decile_cumsum = decile_sum['Sum'].cumsum().reset_index()
        decile_cumsum.columns = ['Decile', 'CumSum']
        decile_cumsum['Decile'] = decile_cumsum['Decile'] + 1
        decile_count = results.groupby('Decile')['Count'].count().reset_index()
        Decile_sum=decile_sum.join(decile_count.set_index('Decile'),on='Decile')
        Decile_sum=Decile_sum.join(decile_cumsum.set_index('Decile'),on='Decile')
        Decile_sum['gain'] = Decile_sum['CumSum']/decile_sum['Sum'].sum()
        Decile_sum['Event Rate']=Decile_sum['Sum']/Decile_sum['Count']
        # if req_dig:
        #     ax = plt.figure(figsize=(12, 8))
        #     plt.title('Decile Score - Cumulative Gain Plot')
        #     sns.lineplot(
        #         x = Decile_sum['Decile'], y=Decile_sum['gain']*100, label='model')
        #     sns.lineplot(x=Decile_sum['Decile'],
        #                 y=Decile_sum['Decile']*10, label='avg')

        #     return Decile, Decile_sum, ax
        # else:
        return Decile, Decile_sum
        
    @staticmethod
    def model_result(test_df, features, target, iftrain = 'Yes', model = None, threshold = None):
        train_result = pd.DataFrame(test_df[target].copy()) 
        y_pred_train =[x[1] for x in model.predict_proba(test_df[features])] # feature needs to be replaced with s_feature
        train_result['Score'] = y_pred_train
        roc_auc, cutoff_df = ModelViz.get_optimum_threshold(train_result, target=target)
        if threshold is None:
            print('Optimal Cutoff: ',cutoff_df['thresholds'].values[0])
            thresholds = cutoff_df['thresholds'].values[0]
            print(f'ROC AUC Score at an Optimum Threshold: \n {cutoff_df} \n Threshold',roc_auc_score(test_df[target],y_pred_train))
        else: 
            thresholds = np.float64(threshold)
        train_pred = np.where(y_pred_train > thresholds, 1, 0)
        class_report = classification_report(test_df[target], train_pred)
        print('Classification Report: \n', class_report)
        if iftrain == 'Yes':
            train_result['Decile'] = 10 - pd.qcut(train_result['Score'], 10, labels=False)
            decile_prob = list(train_result.groupby('Decile')['Score'].min())
            return train_result, decile_prob, thresholds, class_report
        else:
            return train_result, class_report

    # Define the objective function
    def objective(self, params, model_type):
        if model_type == 'lightgbm':
            model = lgb.LGBMClassifier(**params)
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(**params)
        elif model_type == 'randomforest':
            model = RandomForestClassifier(**params)
        elif model_type == 'logistic':
            model = LogisticRegression(**params)
        
        model.fit(self.train[self.s_feature], self.train[self.target])
        preds = model.predict_proba(self.eval[self.s_feature])[:, 1]
        auc = roc_auc_score(self.eval[self.target], preds)
        return {'loss': -auc, 'status': STATUS_OK}

    # Function to optimize and evaluate models
    def optimize_and_evaluate(self):

        # Define the search space for each model
        space_lightgbm = {
            'num_leaves': hp.choice('num_leaves', range(20, 150)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'n_estimators': hp.choice('n_estimators', range(50, 500))
        }

        space_xgboost = {
            'max_depth': hp.choice('max_depth', range(3, 10)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'n_estimators': hp.choice('n_estimators', range(50, 500))
        }

        space_randomforest = {
            'n_estimators': hp.choice('n_estimators', range(50, 500)),
            'max_depth': hp.choice('max_depth', range(3, 20))
        }

        space_logistic = {
            'C': hp.uniform('C', 0.01, 10)
        }

        models = {
            'lightgbm': space_lightgbm,
            'xgboost': space_xgboost,
            'randomforest': space_randomforest,
            'logistic': space_logistic
        }

        results = {}
        
        for model_type, space in models.items():
            trials = Trials()
            best = fmin(fn=lambda params: self.objective(params, model_type),
                        space=space,
                        algo=tpe.suggest,
                        max_evals=50,
                        trials=trials)
            
            model = None
            if model_type == 'lightgbm':
                model = lgb.LGBMClassifier(**best)
            elif model_type == 'xgboost':
                model = xgb.XGBClassifier(**best)
            elif model_type == 'randomforest':
                model = RandomForestClassifier(**best)
            elif model_type == 'logistic':
                model = LogisticRegression(**best)
            
            # Fitting the model with the best parameters
            model.fit(self.train[self.s_feature], self.train[self.target])

            # Getting feature importance
            if model_type != 'logistic':
                importance = model.feature_importances_
            else:
                # Getting feature importance
                model_coeff = model.coef_

                # Creating a DataFrame for better visualization
                importance = pd.DataFrame({
                    'Feature': self.s_feature,
                    'Importance': model_coeff[0]
                }).sort_values(by='Importance', ascending=False)

            # Train Result
            train_result, decile_prob, threshold, train_class_report = self.model_result(self.train, self.s_feature, self.target, iftrain = 'Yes', model = model, threshold = None)
            training_decile, train_decile_summary = self.decile_summary(prob = train_result['Score'], actual = train_result[self.target], prob_list = decile_prob)
            
            # Test Result
            eval_result, eval_class_report = self.model_result(self.eval, self.s_feature, self.target, iftrain = 'No', model = model, threshold = threshold)
            eval_decile, eval_decile_summary = self.decile_summary(prob = eval_result['Score'], actual = eval_result[self.target], prob_list = decile_prob)

            if self.validation is None:
                results[model_type] = [model, train_decile_summary, eval_decile_summary, train_class_report, eval_class_report]
            else:
                # Test Result
                valid_result, valid_class_report = self.model_result(self.eval, self.s_feature, self.target, iftrain = 'No', model = model, threshold = threshold)
                valid_decile, valid_decile_summary = self.decile_summary(prob = eval_result['Score'], actual = eval_result[self.target], prob_list = decile_prob, req_dig=False)
                results[model_type] = [train_decile_summary, eval_decile_summary, valid_decile_summary, train_class_report, eval_class_report, valid_class_report]
        print('Model run and evaluation is completed')
        return results

master_file_path = '../Data/Master Variables.xlsx'

# obj = DataGenerator(master_file_path=master_file_path, n_samples=10000)
# df = obj.generate_dataset()
obj = data_preprocess(master_file_path = master_file_path, 
                      input_data_path = None, 
                      target = 'li_flag', 
                      test_split=0.3, 
                      generate_random_data=True, 
                      n_samples = 10000)
train, test, s_feature = obj.run_pre_processing()

# Example usage
model_obj = model_evaluate(train = train, eval = test, target = 'li_flag', s_feature = s_feature)
result = model_obj.optimize_and_evaluate()
