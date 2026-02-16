"""
01 — Build Analytic Dataset (Instacart)

Generated from course notebooks via tools/merge_course_notebooks.py
"""


#----------------------------------------------------------------------
# Load Raw Data
#----------------------------------------------------------------------

# Import libraries
import pandas as pd
import numpy as np
import os

path = r'/Users/spencer/Documents/Career Foundry/Data Immersion/4 Python Fundamentals for Data Analysts/Instacart Basket Analysis'

vars_list = ['order_id', 'user_id', 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']

df = pd.read_csv(os.path.join(path, '02 Data', 'Original Data', 'orders.csv'), usecols = vars_list)

df.shape

df.head()

df_prods = pd.read_csv(os.path.join(path, '02 Data', 'Original Data', 'products.csv'), index_col = False)

df_prods.head(20)

df_prods.tail(35)

df_prods.columns

df_prods.shape

df_prods.describe()

df_prods.info()


#----------------------------------------------------------------------
# Data Quality Checks
#----------------------------------------------------------------------

# Import libraries
import pandas as pd
import numpy as np
import os

# Import datasets
path = r'/Users/spencer/Documents/Career Foundry/Data Immersion/4 Python Fundamentals for Data Analysts/Instacart Basket Analysis'
df_ords = pd.read_csv(os.path.join(path, '02 Data', 'Prepared Data', 'orders_wrangled.csv'), index_col = False)
df_prods = pd.read_csv(os.path.join(path, '02 Data', 'Original Data', 'products.csv'), index_col = False)
df_dep = pd.read_csv(os.path.join(path, '02 Data', 'Prepared Data', 'departments_wrangled.csv'), index_col = False)

# Create a dataframe

df_test = pd.DataFrame()

# Create a mixed type column

df_test['mix'] = ['a', 'b', 1, True]

df_test.head()

for col in df_test.columns.tolist():
  weird = (df_test[[col]].map(type) != df_test[[col]].iloc[0].apply(type)).any(axis = 1)
  if len (df_test[weird]) > 0:
    print (col)

df_test['mix'] = df_test['mix'].astype('str')

df_prods.isnull().sum()

df_nan = df_prods[df_prods['product_name'].isnull() == True]

df_nan

df_prods.shape

# Make new dataframe, removing null 'product_name' values
df_prods_clean = df_prods[df_prods['product_name'].isnull() == False]

df_prods_clean.shape

df_dups = df_prods_clean[df_prods_clean.duplicated()]

df_dups

df_prods_clean.shape

df_prods_clean_no_dups = df_prods_clean.drop_duplicates()

df_prods_clean_no_dups.shape

df_ords.describe()

# Sanity check to make sure it wasnt a trick question?
df_ords.head(10)

for col in df_ords.columns.tolist():
  weird = (df_ords[[col]].map(type) != df_ords[[col]].iloc[0].apply(type)).any(axis = 1)
  if len (df_ords[weird]) > 0:
    print (col)

df_ords.isnull().sum()

# Create a new column 'first_order' if 'days_since_prior_order' is null
df_ords['first_order'] = df_ords['days_since_prior_order'].isnull()

# Check work
df_ords.head()

df_ords_dups = df_ords[df_ords.duplicated()]

df_ords_dups

df_prods_clean_no_dups.to_csv(os.path.join(path, '02 Data','Prepared Data', 'products_checked.csv'), index=False)

df_ords.to_csv(os.path.join(path, '02 Data','Prepared Data', 'orders_checked.csv'), index=False)


#----------------------------------------------------------------------
# Merge Tables
#----------------------------------------------------------------------

# Import libraries
import pandas as pd
import numpy as np
import os

# Import datasets
path = r'/Users/spencer/Documents/Career Foundry/Data Immersion/4 Python Fundamentals for Data Analysts/Instacart Basket Analysis'
df_ords = pd.read_csv(os.path.join(path, '02 Data', 'Prepared Data', 'orders_checked.csv'), index_col = False)
df_prods = pd.read_csv(os.path.join(path, '02 Data', 'Prepared Data', 'products_checked.csv'), index_col = False)
df_dep = pd.read_csv(os.path.join(path, '02 Data', 'Prepared Data', 'departments_wrangled.csv'), index_col = False)
df_ords_prior = pd.read_csv(os.path.join(path, '02 Data', 'Original Data', 'orders_products_prior.csv'), index_col = False)

df_ords_prior.head()

df_ords.head()

df_ords_prior.shape

df_ords.shape

df_merged_large = df_ords.merge(df_ords_prior, on = 'order_id', indicator = True)

# check output
df_merged_large.head()

df_merged_large.shape

df_merged_large['_merge'].value_counts()

# drop indicator flag (_merge)
df_merged_large = df_merged_large.drop(columns = ['_merge'])

df_merged_large.head()

df_merged_large.to_pickle(os.path.join(path, '02 Data','Prepared Data', 'orders_products_combined.pkl'))


#----------------------------------------------------------------------
# Merge Validation & Export
#----------------------------------------------------------------------

# Import libraries
import pandas as pd
import numpy as np
import os

# Import datasets
path = r'/Users/spencer/Documents/Career Foundry/Data Immersion/4 Python Fundamentals for Data Analysts/Instacart Basket Analysis'
df_ords = pd.read_csv(os.path.join(path, '02 Data', 'Prepared Data', 'orders_checked.csv'), index_col = False)
df_prods = pd.read_csv(os.path.join(path, '02 Data', 'Prepared Data', 'products_checked.csv'), index_col = False)
df_dep = pd.read_csv(os.path.join(path, '02 Data', 'Prepared Data', 'departments_wrangled.csv'), index_col = False)
df_ords_prior = pd.read_csv(os.path.join(path, '02 Data', 'Original Data', 'orders_products_prior.csv'), index_col = False)
df_ords_prods_comb = pd.read_pickle(os.path.join(path, '02 Data', 'Prepared Data', 'orders_products_combined.pkl'))

# verify shape
df_ords_prods_comb.shape

df_ords_prods_comb.head()

df_prods.head()

df_prods.shape

print(df_prods[df_prods.duplicated()])

df_dups = df_prods[df_prods.duplicated(subset=['product_id'])]
print(df_dups)

df_prods = df_prods.drop_duplicates(subset=['product_id'])

df_prods.shape

df_merged = df_ords_prods_comb.merge(df_prods, on = 'product_id', how = 'left', indicator = True)

df_merged['_merge'].value_counts()

# expect 32434489 rows
df_merged.shape

df_merged = df_merged.drop(columns = ['_merge'])

df_merged.head()

df_merged.to_pickle(os.path.join(path, '02 Data','Prepared Data', 'ords_prods_merge.pkl'))
