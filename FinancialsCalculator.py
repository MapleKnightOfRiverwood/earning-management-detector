import statsmodels.api as sm
import warnings
import pandas as pd
import wrds

warnings.filterwarnings('ignore')

class FinancialsCalculator:
    def __inti__(self):
        pass

    @staticmethod
    def calculateFinancials(db, tic, startYear, endYear):

        startYear = startYear - 1  # Since we are calculating changes, we need the previous year's data as well

        # Query the data from Compustat
        query = f"""
        SELECT fyear, ni, oancf, revt, rect, at, ppegt, act, lct, lt, ebit, epspx, dp
        FROM compa.funda
        WHERE tic = '{tic}'
        AND indfmt = 'INDL'
        AND datafmt = 'STD'
        AND popsrc = 'D'
        AND consol = 'C'
        AND fyear BETWEEN '{startYear}' AND '{endYear}'
        ORDER BY fyear
        """

        financial_data = db.raw_sql(query)

        # Check the validity of the company's financial data
        if len(financial_data) != endYear - startYear + 1:  # Return error if company does not exist in all years
            print('Financial data length is ' + str(len(financial_data)))
            return 1
        if financial_data.isna().sum().sum() != 0:  # Return error if company has missing financial information
            print('Company has ' + str(financial_data.isna().sum().sum()) + ' missing data')
            return 1
        if 0 in financial_data['at'].values:  # Return error if company has 0 in total asset
            print('Company total asset is 0')
            return 1
        print('financial data is valid')

        # Calculate accruals and lagged assets
        financial_data['ta'] = financial_data['ni'] - financial_data['oancf']
        financial_data['lagged_assets'] = financial_data['at'].shift(1)

        # Calculate changes in revenue and accounts receivable
        financial_data['delta_rev'] = financial_data['revt'].diff()
        financial_data['delta_ar'] = financial_data['rect'].diff()

        # Modified jones model variables
        financial_data['inv_lagged_assets'] = 1 / financial_data['lagged_assets']
        financial_data['scaled_delta_rev'] = (financial_data['delta_rev'] - financial_data['delta_ar']) / \
                                             financial_data['lagged_assets']
        financial_data['ppe_scaled'] = financial_data['ppegt'] / financial_data['lagged_assets']

        # Teoh et al. model variables
        financial_data['tca'] = financial_data['ta'] + financial_data['dp']  # Add depreciation to total accruals to get total current accruals

        # Dechow and Dichev Model variables - This is a modified Dechow and Dichev Model. The CFO based Dechow and
        # Dichev Model require the use of CFO t+1 which is not available at year t, so it cannot be used to
        # evaluate the current year's financial statement risk.
        financial_data['chg_wc'] = (financial_data['act'] - financial_data['lct']) - (
                    financial_data['act'].shift(1) - financial_data['lct'].shift(1))
        financial_data['scaled_chg_wc'] = financial_data['chg_wc'] / financial_data['lagged_assets']
        financial_data['scaled_ebit'] = financial_data['ebit'] / financial_data['lagged_assets']

        # Drop missing values (First line will be dropped since we are calculating changes)
        financial_data = financial_data.dropna()

        # Estimate the modified jones model
        y = financial_data['ta'] / financial_data['lagged_assets']
        X = financial_data[['inv_lagged_assets', 'scaled_delta_rev', 'ppe_scaled']]
        X = sm.add_constant(X)

        model = sm.OLS(y, X)
        results = model.fit()

        # Calculate the UAA scores based on the modified jones model
        financial_data['uaa_mj'] = results.resid.abs()

        # Estimate the Teoh et al. Model
        y_teoh = financial_data['tca'] / financial_data['lagged_assets']
        X_teoh = financial_data[['inv_lagged_assets', 'scaled_delta_rev']]
        X_teoh = sm.add_constant(X_teoh)

        model_teoh = sm.OLS(y_teoh, X_teoh)
        results_teoh = model_teoh.fit()

        # Calculate the UAA scores based on the teoh model
        financial_data['uaa_teoh'] = results_teoh.resid.abs()

        # Estimate the Dechow and Dichev Model
        y_dd = financial_data['ta'] / financial_data['lagged_assets']
        X_dd = financial_data[['inv_lagged_assets', 'scaled_chg_wc', 'scaled_ebit', 'scaled_delta_rev']]
        X_dd = sm.add_constant(X_dd)

        model_dd = sm.OLS(y_dd, X_dd)
        results_dd = model_dd.fit()

        # Calculate the residuals (Unexpected Accruals) for the Dechow and Dichev Model
        financial_data['uaa_dd'] = results_dd.resid.abs()

        # Calculate Shareholder's Equity
        financial_data['shareholders_equity'] = financial_data['at'] - financial_data['lt']

        # Calculate current ratio
        financial_data['current_ratio'] = financial_data['act'] / financial_data['lct']

        # Calculate debt to equity
        financial_data['debt_to_equity'] = financial_data['lt'] / financial_data['shareholders_equity']

        # Calculate return on equity (ROE)
        financial_data['roe'] = financial_data['ni'] / financial_data['shareholders_equity']

        # Calculate Asset Turnover
        financial_data['asset_turnover'] = financial_data['revt'] / financial_data['at']

        # Add the percentage change of stock price in each year

        # First query the stock price info from another datatable
        stock_price_query = f"""
        SELECT datadate, prccm
        FROM compa.secm
        WHERE tic = '{tic}'
        AND EXTRACT(MONTH FROM datadate) = 12
        AND datadate BETWEEN '{startYear}-12-01' AND '{endYear}-12-31'
        ORDER BY datadate
        """

        stock_price_data = db.raw_sql(stock_price_query)

        # Check the validity of the company's financial data
        if len(stock_price_data) != endYear - startYear + 1:  # Return error if company does not exist in all years
            print('Stock data length is ' + str(len(stock_price_data)))
            return 1

        # Convert datadate to datetime and set it as index
        stock_price_data['datadate'] = pd.to_datetime(stock_price_data['datadate'])
        stock_price_data.set_index('datadate', inplace=True)

        # Calculate the annual stock price percentage change
        stock_price_data['pct_change_stock'] = stock_price_data['prccm'].pct_change().mul(100)

        # Add the fyear column to the stock price data
        stock_price_data['fyear'] = stock_price_data.index.year

        # Merge the financial data and stock price data on fyear
        financial_data = financial_data.merge(stock_price_data[['fyear', 'prccm', 'pct_change_stock']], on='fyear', how='left')

        # Calculate earnings yield
        financial_data['earnings_yield'] = financial_data['epspx'] / financial_data['prccm']

        returnDF = financial_data[['uaa_mj', 'uaa_teoh', 'uaa_dd', 'pct_change_stock', 'earnings_yield', 'current_ratio', 'debt_to_equity',
                                   'roe', 'asset_turnover']]

        # Return all the scores and ratios
        return returnDF





'''
# Reserved for testing
import statsmodels.api as sm
import warnings
import wrds
import pandas as pd
warnings.filterwarnings('ignore')

db = wrds.Connection(wrds_username='david402')

startYear = 2012
endYear = 2021
tic = 'GOOGL'

startYear = startYear - 1  # Since we are calculating changes, we need the previous year's data as well

# Query the data from Compustat
query = f"""
SELECT fyear, ni, oancf, revt, rect, at, ppegt, act, lct, lt, ebit, epspx, dp
FROM compa.funda
WHERE tic = '{tic}'
AND indfmt = 'INDL'
AND datafmt = 'STD'
AND popsrc = 'D'
AND consol = 'C'
AND fyear BETWEEN '{startYear}' AND '{endYear}'
ORDER BY fyear
"""

financial_data = db.raw_sql(query)


# Check the validity of the company's financial data
if len(financial_data) != endYear - startYear + 1:  # Return error if company does not exist in all years
    print('Financial data length is ' + str(len(financial_data)))
if financial_data.isna().sum().sum() != 0:  # Return error if company has missing financial information
    print('Company has ' + str(financial_data.isna().sum().sum()) + ' missing data')
if 0 in financial_data['at'].values:  # Return error if company has 0 in total asset
    print('Company total asset is 0')


# Calculate accruals and lagged assets
financial_data['ta'] = financial_data['ni'] - financial_data['oancf']
financial_data['lagged_assets'] = financial_data['at'].shift(1)

# Calculate changes in revenue and accounts receivable
financial_data['delta_rev'] = financial_data['revt'].diff()
financial_data['delta_ar'] = financial_data['rect'].diff()

# Modified jones model variables
financial_data['inv_lagged_assets'] = 1 / financial_data['lagged_assets']
financial_data['scaled_delta_rev'] = (financial_data['delta_rev'] - financial_data['delta_ar']) / financial_data['lagged_assets']
financial_data['ppe_scaled'] = financial_data['ppegt'] / financial_data['lagged_assets']

# Teoh et al. model variables
financial_data['tca'] = financial_data['ta'] + financial_data['dp']  # Add depreciation to total accruals to get total current accruals

# Dechow and Dichev Model variables
financial_data['chg_wc'] = (financial_data['act'] - financial_data['lct']) - (financial_data['act'].shift(1) - financial_data['lct'].shift(1))
financial_data['scaled_chg_wc'] = financial_data['chg_wc'] / financial_data['lagged_assets']
financial_data['scaled_ebit'] = financial_data['ebit'] / financial_data['lagged_assets']

# Drop missing values (First line will be dropped since we are calculating changes)
financial_data = financial_data.dropna()

# Estimate the modified jones model
y = financial_data['ta'] / financial_data['lagged_assets']
X = financial_data[['inv_lagged_assets', 'scaled_delta_rev', 'ppe_scaled']]
X = sm.add_constant(X)

model = sm.OLS(y, X)
results = model.fit()

# Calculate the UAA scores based on the modified jones model
financial_data['uaa_mj'] = results.resid.abs()

# Estimate the Teoh et al. Model
y_teoh = financial_data['tca'] / financial_data['lagged_assets']
X_teoh = financial_data[['inv_lagged_assets', 'scaled_delta_rev']]
X_teoh = sm.add_constant(X_teoh)

model_teoh = sm.OLS(y_teoh, X_teoh)
results_teoh = model_teoh.fit()

# Calculate the UAA scores based on the teoh model
financial_data['uaa_teoh'] = results_teoh.resid.abs()

# Estimate the Dechow and Dichev Model
y_dd = financial_data['ta'] / financial_data['lagged_assets']
X_dd = financial_data[['inv_lagged_assets', 'scaled_chg_wc', 'scaled_ebit', 'scaled_delta_rev']]
X_dd = sm.add_constant(X_dd)

model_dd = sm.OLS(y_dd, X_dd)
results_dd = model_dd.fit()

# Calculate the residuals (Unexpected Accruals) for the Dechow and Dichev Model
financial_data['uaa_dd'] = results_dd.resid.abs()

# Calculate Shareholder's Equity
financial_data['shareholders_equity'] = financial_data['at'] - financial_data['lt']

# Calculate current ratio
financial_data['current_ratio'] = financial_data['act'] / financial_data['lct']

# Calculate debt to equity ratio
financial_data['debt_equity_ratio'] = financial_data['lt'] / financial_data['shareholders_equity']

# Calculate return on equity (ROE)
financial_data['roe'] = financial_data['ni'] / financial_data['shareholders_equity']

# Calculate Asset Turnover
financial_data['asset_turnover'] = financial_data['revt'] / financial_data['at']

# Add the percentage change of stock price in each year
stock_price_query = f"""
SELECT datadate, prccm
FROM compa.secm
WHERE tic = '{tic}'
AND EXTRACT(MONTH FROM datadate) = 12
AND datadate BETWEEN '{startYear}-12-01' AND '{endYear}-12-31'
ORDER BY datadate
"""

stock_price_data = db.raw_sql(stock_price_query)

if len(stock_price_data) != endYear - startYear + 1:  # Return error if company does not exist in all years
    print('Stock data length is ' + str(len(stock_price_data)))

# Convert datadate to datetime and set it as index
stock_price_data['datadate'] = pd.to_datetime(stock_price_data['datadate'])
stock_price_data.set_index('datadate', inplace=True)

# Calculate the annual stock price percentage change
stock_price_data['pct_change_stock'] = stock_price_data['prccm'].pct_change().mul(100)

# Add the fyear column to the stock price data
stock_price_data['fyear'] = stock_price_data.index.year

# Merge the financial data and stock price data on fyear
financial_data = financial_data.merge(stock_price_data[['fyear', 'prccm', 'pct_change_stock']], on='fyear', how='left')

# Calculate earnings yield
financial_data['earnings_yield'] = financial_data['epspx'] / financial_data['prccm']
'''



