import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

class CarPrice:

    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
        print(f'${len(self.df)} lines loaded')

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    def exploratory_data_analysis(self):
        plt.figure(figsize=(6, 4))
        sns.distplot(self.df.msrp, kde=False, hist_kws=dict(color='black', alpha=1))
        plt.ylabel('Frequency')
        plt.xlabel('Price')
        plt.title('Distribution of prices')
        plt.show()
        plt.figure(figsize=(6, 4))
        sns.distplot(self.df.msrp[self.df.msrp < 100000], kde=False, hist_kws=dict(color='black', alpha=1))
        plt.ylabel('Frequency')
        plt.xlabel('Price')
        plt.title('Distribution of prices')
        plt.show()
        log_price = np.log1p(self.df.msrp)
        plt.figure(figsize=(6, 4))
        sns.distplot(log_price, kde=False, hist_kws=dict(color='black', alpha=1))
        plt.ylabel('Frequency')
        plt.xlabel('Log(Price + 1)')
        plt.title('Distribution of prices after log tranformation')
        plt.show()
        self.df.isnull().sum()

    def validate(self):
        np.random.seed(2)
        n = len(self.df)
        n_val = int(0.2 * n)
        n_test = int(0.2 * n)
        n_train = n - (n_val + n_test)
        idx = np.arange(n)
        np.random.shuffle(idx)
        df_shuffled = self.df.iloc[idx]
        df_train = df_shuffled.iloc[:n_train].copy()
        df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
        df_test = df_shuffled.iloc[n_train+n_val:].copy()
        y_train_orig = df_train.msrp.values
        y_val_orig = df_val.msrp.values
        y_test_orig = df_test.msrp.values
        y_train = np.log1p(df_train.msrp.values)
        y_val = np.log1p(df_val.msrp.values)
        y_test = np.log1p(df_test.msrp.values)
        del df_train['msrp']
        del df_val['msrp']
        del df_test['msrp']
        return df_train, df_test, df_val, y_val_orig, y_train, y_test, y_val

    def linear_regression(self, X, y):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])
        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)
        return w[0], w[1:]

    def prepare_X(self, df):
        base = ['engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
        # base = ['transmission_type', 'driven_wheels', 'number_of_doors', 'market_category', 'vehicle_size', 'vehicle_style']
        features = base.copy()
        df = df.copy()

        # df['age'] = 2017 - df.year
        # features.append('age')
        
        for v in [2, 3, 4]:
            feature = 'num_doors_%s' % v
            df[feature] = (df['number_of_doors'] == v).astype(int)
            features.append(feature)

        # for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
        #     feature = 'is_make_%s' % v
        #     df[feature] = (df['make'] == v).astype(int)
        #     features.append(feature)

        # for v in ['regular_unleaded', 'premium_unleaded_(required)', 
        #         'premium_unleaded_(recommended)', 'flex-fuel_(unleaded/e85)']:
        #     feature = 'is_type_%s' % v
        #     df[feature] = (df['engine_fuel_type'] == v).astype(int)
        #     features.append(feature)

        for v in ['automatic', 'manual', 'automated_manual']:
            feature = 'is_transmission_%s' % v
            df[feature] = (df['transmission_type'] == v).astype(int)
            features.append(feature)

        for v in ['front_wheel_drive', 'rear_wheel_drive', 'all_wheel_drive', 'four_wheel_drive']:
            feature = 'is_driven_wheens_%s' % v
            df[feature] = (df['driven_wheels'] == v).astype(int)
            features.append(feature)

        for v in ['crossover', 'flex_fuel', 'luxury', 'luxury,performance', 'hatchback']:
            feature = 'is_mc_%s' % v
            df[feature] = (df['market_category'] == v).astype(int)
            features.append(feature)

        for v in ['compact', 'midsize', 'large']:
            feature = 'is_size_%s' % v
            df[feature] = (df['vehicle_size'] == v).astype(int)
            features.append(feature)

        for v in ['sedan', '4dr_suv', 'coupe', 'convertible', '4dr_hatchback']:
            feature = 'is_style_%s' % v
            df[feature] = (df['vehicle_style'] == v).astype(int)
            features.append(feature)

        df_num = df[features]
        df_num = df_num.fillna(0)
        X = df_num.values
        return X

    def rmse(self, y, y_pred):
        error = y_pred - y
        mse = (error ** 2).mean()
        return np.sqrt(mse)

def test() -> None:
    carPrice = CarPrice()
    carPrice.trim()
    # carPrice.exploratory_data_analysis()
    df_train, df_test, df_val, y_val_orig, y_train, y_test, y_val = carPrice.validate()
    X_train = carPrice.prepare_X(df_train)
    w_0, w = carPrice.linear_regression(X_train, y_train)
    print(w_0)
    print(w)
    y_pred = w_0 + X_train.dot(w)
    print('train:', carPrice.rmse(y_train, y_pred))

    X_val = carPrice.prepare_X(df_val)
    y_pred = w_0 + X_val.dot(w)
    print('validation:', carPrice.rmse(y_val, y_pred))
    
    rst = pd.DataFrame(df_val[:5], columns=['engine_cylinders', 'transmission_type', 'driven_wheels', 'number_of_doors', 'market_category', 'vehicle_size', 'vehicle_style', 'highway_mpg', 'city_mpg', 'popularity'])
    print(rst)
    # temp1 = pd.DataFrame(np.expm1(y_pred[:5]), columns=['msrp'])
    # temp2 = pd.DataFrame(y_val_orig[:5], columns=['msrp_pred'])
    temp1 = np.expm1(y_pred[:5])
    temp2 = y_val_orig[:5]
    rst['msrp'] = temp1
    rst['msrp_pred'] = temp2
    pd.set_option("display.max_column", None)
    print(rst)

if __name__ == "__main__":
    test()