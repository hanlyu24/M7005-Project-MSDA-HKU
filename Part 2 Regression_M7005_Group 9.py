import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# read the combined Data
data = pd.read_excel('C:/Users/86156/Desktop/Combined Data.xlsx')

# 1. Data Standardization
# Standardizing the education, income, and occupation variables
scaler = StandardScaler()
data[['Level of Education', 'Family Income', 'Work']] = scaler.fit_transform(data[['Level of Education', 'Family Income', 'Work']])

# 2.Generate latent socioeconomic status (SES) using Factor Analysis
# Extract socioeconomic status (SES) using factor analysis
factor = FactorAnalysis(n_components=1)  # (Extract one factor)
data['SES'] = factor.fit_transform(data[['Level of Education', 'Family Income', 'Work']])


# Transform SES into high and low socioeconomic status
median_ses = data['SES'].median()  # (Using the median to split)
data['SES_high_low'] = np.where(data['SES'] > median_ses, 1, 0)  # (1 for high SES, 0 for low SES)

# 3. OLS regression analysis
# Using SES as the independent variable, depression as the dependent variable, and other variables as control variables
# Control variables are: gender, marriage, and religion
X = data[['SES_high_low', 'Gender', 'Marriage', 'Religious Belief']]  # (Control variables)
y = data['Depression']  #(Dependent variable: depression)


# Adding a constant (intercept) to the independent variables X
X = sm.add_constant(X)


# Performing OLS regression analysis
ols_model = sm.OLS(y, X)  #  (OLS regression model)
result = ols_model.fit()  #  (Fit the model)


# Output the regression results
print(result.summary())  #  (Print the regression summary)

