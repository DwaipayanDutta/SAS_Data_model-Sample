import numpy as np
import pandas as pd
np.random.seed(42)

class ContinuousDataGenerator:
    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.cont_dict = {
            'Variable': [
                'AGE', 'AQB_BALANCE', 'AUM', 'CIBIL_SCORE', 'CREDIT_CARD_LIMIT',
                'CR_AMT_12MNTH', 'CR_CNT_12MNTH', 'DC_APPAREL_30DAYS_ACTV', 'DC_ECOM_30DAYS_AMT',
                'DC_ECOM_30DAYS_CNT', 'DC_FOOD_30DAYS_ACTV', 'DC_FUEL_30DAYS_ACTV', 'DC_GROCRY_30DAYS_ACTV',
                'DC_OTT_30DAYS_ACTV', 'DC_POS_30DAYS_AMT', 'DC_POS_30DAYS_CNT', 'DC_RECHARGE_30DAYS_ACTV',
                'DC_TRAVEL_30DAYS_ACTV', 'DC_UTILITY_30DAYS_ACTV', 'DR_AMT_12MNTH', 'DR_CNT_12MNTH',
                'DR_CR_RATIO', 'FD_COUNT', 'FD_CURRENTMONTHANR', 'FD_FLAG', 'GI_FLAG', 'HEALTH_FLAG',
                'INCOME_NET', 'KYC_LAST_DONE_DATE', 'LI_FLAG', 'MASS_FLAG', 'MBK_ACTIVE', 'MF_FLAG',
                'MONTHLY_BALANCE', 'NRV', 'NR_FLAG', 'TOTAL_LIVE_SECURED_AMT', 'TOTAL_LIVE_UNSECURED_AMT',
                'VINTAGE_DAYS'
            ],
            'Mean': [
                45.75, 335324.5275, 1482679.66, 729.25, 432750.0, 4427092.8125, 390.0, 0.0, 747.54, 0.5, 0.0, 0.25,
                0.25, 0.0, 1250.0, 0.75, 0.0, 0.0, 0.0, 4400780.205, 755.25, 18.4664255, 2.25, 818797.965, 0.5, 0.25,
                0.0, 637578.845, 22694.0, 0.25, 1.0, 0.5, 0.25, 285443.28500000003, 1427096.995, 0.25, 8043387.25,
                1626783.5, 2472.25
            ],
            'Std Dev': [
                13.2034277, 40797470.12, 31660895.68, 145.8168632, 227937.36, 160056333.0, 405.0004491, 0.0778686,
                55320.12, 1.1421999, 0.0979862, 0.1093538, 0.1182655, 0.081972, 6117.33, 1.1075376, 0.0678652, 0.0594367,
                0.0595447, 163060150.0, 435.6866233, 249.0156422, 1.8768659, 2177717.88, 0.3077795, 0.1006145, 0.0901247,
                272956.6, 786.2686886, 0.1785988, 0.1920778, 0.4817677, 0.1128338, 39394146.0, 31877424.32, 0.1684775,
                24655099.6, 3300140.98, 1216.33
            ]
        }

    def generate_data(self):
        data_summary = pd.DataFrame(self.cont_dict)
        df_cont = []
        for _, row in data_summary.iterrows():
            sample = np.random.normal(loc=row['Mean'], scale=row['Std Dev'], size=self.n_samples)
            sample = np.clip(sample, a_min=0, a_max=None)
            col_name = row['Variable']
            sample_data = pd.DataFrame({col_name: sample})
            df_cont.append(sample_data)
        return pd.concat(df_cont, axis=1)


class CategoricalDataGenerator:
    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.cat_variables = {
            'ACCOUNT_TYPE': ['SAVINGS', 'CURRENT'],
            'CUSTOMER_TAG': ['RETAIL', 'INSTITUTIONAL'],
            'EDUCATION_LEVEL': ['Below 10th', 'UG', 'Graduate', 'PG'],
            'GENDER': ['MALE', 'FEMALE'],
            'INTERNET_BANKING_USAGE': ['Y', 'N'],
            'MARITAL_STATUS': ['Married', 'Single/Unmarried', 'Divorced/Separated'],
            'NOMINEE_AVAILABLE_FLAG': ['Y', 'N'],
            'RM_ALLOCATED_FLAG': ['Y', 'N'],
            'OCCUPATION': ['SELF EMPLOYED', 'SALARIED', 'BUSINESS OWNER', 'NON-EARNER'],
            'STATE': [
                'Maharashtra', 'Gujarat', 'West Bengal', 'Uttar Pradesh', 'Tamil Nadu', 'Haryana', 'Rajasthan',
                'Punjab', 'Delhi', 'Madhya Pradesh', 'Out of India', 'Andhra Pradesh', 'Karnataka', 'Telangana',
                'Odisha', 'Kerala', 'Bihar', 'Assam', 'Chhattisgarh', 'Jharkhand', 'Uttarakhand', 'Himachal Pradesh',
                'Jammu and Kashmir', 'Chandigarh', 'Goa', 'Manipur', 'Nagaland', 'Tripura', 'Puducherry',
                'Dadra and Nagar Haveli and Daman and Diu', 'Sikkim', 'Arunachal Pradesh', 'Meghalaya', 'Mizoram',
                'Andaman and Nicobar'
            ]
        }
        self.prob = {
            'ACCOUNT_TYPE': [0.80, 0.20],
            'CUSTOMER_TAG': [0.5, 0.5],
            'EDUCATION_LEVEL': [0.25, 0.25, 0.25, 0.25],
            'GENDER': [0.6, 0.4],
            'INTERNET_BANKING_USAGE': [0.4, 0.6],
            'MARITAL_STATUS': [0.6, 0.35, 0.05],
            'NOMINEE_AVAILABLE_FLAG': [0.95, 0.05],
            'RM_ALLOCATED_FLAG': [0.98, 0.02],
            'OCCUPATION': [0.404, 0.339, 0.152, 0.105],
            'STATE': [
                0.119496018, 0.089888389, 0.07827775, 0.072670547, 0.064865747, 0.062475027, 0.050854399,
                0.049988679, 0.046685624, 0.043482459, 0.036553368, 0.033316907, 0.03284409, 0.031895128,
                0.030473349, 0.029424496, 0.028928372, 0.020664074, 0.018809435, 0.015855997, 0.007232093,
                0.006562825, 0.005510642, 0.005430729, 0.005330838, 0.00238739, 0.001664846, 0.001534988,
                0.001481713, 0.001328547, 0.001288591, 0.001175382, 0.00107882, 0.000526091, 1.66485E-05
            ]
        }

    def generate_data(self):
        data_categorical = {}
        for variable, categories in self.cat_variables.items():
            data_categorical[variable] = np.random.choice(categories, self.n_samples, p=self.prob[variable])
        return pd.DataFrame(data_categorical)


class DataProcessor:
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def create_final_dataset(self):
        cont_generator = ContinuousDataGenerator(self.n_samples)
        data_cont = cont_generator.generate_data()
        cat_generator = CategoricalDataGenerator(self.n_samples)
        data_cat = cat_generator.generate_data()
        data = data_cat.reset_index()
        data.rename(columns={'index': 'ACCOUNT_NO'}, inplace=True)
        data_final = pd.concat([data, data_cont], axis=1)
        return data_final


# Main execution
if __name__ == "__main__":
    # Number of samples
    N = 10000
    processor = DataProcessor(N)
    data = processor.create_final_dataset()
    print(data.head())
