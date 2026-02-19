"""
01 — Build Analytic Dataset (Instacart)

Generated from course notebooks via tools/merge_course_notebooks.py
"""


#----------------------------------------------------------------------
# Load Raw Data
#----------------------------------------------------------------------

# Import libraries
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Base project path (repo root)
# This file lives in /src, so repo root is one level up.
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / '02 Data'
RAW_DIR = DATA_DIR / 'Original Data'
PREP_DIR = DATA_DIR / 'Prepared Data'

# Make sure prepared data folder exists
PREP_DIR.mkdir(parents=True, exist_ok=True)

vars_list = [
  'order_id',
  'user_id',
  'order_number',
  'order_dow',
  'order_hour_of_day',
  'days_since_prior_order'
]

df = pd.read_csv(RAW_DIR / 'orders.csv', usecols = vars_list)

print(df.shape)

print(df.head())

df_prods = pd.read_csv(RAW_DIR / 'products.csv', index_col = False)

print(df_prods.head(20))

print(df_prods.tail(35))

print(df_prods.columns)

print(df_prods.shape)

print(df_prods.describe())

print(df_prods.info())


#----------------------------------------------------------------------
# Data Quality Checks
#----------------------------------------------------------------------

# Import datasets
df_ords = pd.read_csv(PREP_DIR / 'orders_wrangled.csv', index_col=False)
df_prods = pd.read_csv(RAW_DIR / 'products.csv', index_col=False)
df_dep = pd.read_csv(PREP_DIR / 'departments_wrangled.csv', index_col=False)

# Create a dataframe

df_test = pd.DataFrame()

# Create a mixed type column

df_test['mix'] = ['a', 'b', 1, True]

print(df_test.head())

for col in df_test.columns.tolist():
  weird = (df_test[[col]].map(type) != df_test[[col]].iloc[0].apply(type)).any(axis = 1)
  if len (df_test[weird]) > 0:
    print (col)

df_test['mix'] = df_test['mix'].astype('str')

df_prods.isnull().sum()

df_nan = df_prods[df_prods['product_name'].isnull() == True]

print(df_nan)

print(df_prods.shape)

# Make new dataframe, removing null 'product_name' values
df_prods_clean = df_prods[df_prods['product_name'].isnull() == False]

print(df_prods_clean.shape)

df_dups = df_prods_clean[df_prods_clean.duplicated()]

print(df_dups)

print(df_prods_clean.shape)

df_prods_clean_no_dups = df_prods_clean.drop_duplicates()

print(df_prods_clean_no_dups.shape)

print(df_ords.describe())

print(df_ords.head(10))

for col in df_ords.columns.tolist():
  weird = (df_ords[[col]].map(type) != df_ords[[col]].iloc[0].apply(type)).any(axis = 1)
  if len (df_ords[weird]) > 0:
    print (col)

print(df_ords.isnull().sum())

# Create a new column 'first_order' if 'days_since_prior_order' is null
df_ords['first_order'] = df_ords['days_since_prior_order'].isnull()

# Check work
print(df_ords.head())

df_ords_dups = df_ords[df_ords.duplicated()]

print(df_ords_dups)

df_prods_clean_no_dups.to_csv(PREP_DIR / 'products_checked.csv', index=False)
df_ords.to_csv(PREP_DIR / 'orders_checked.csv', index=False)


#----------------------------------------------------------------------
# Merge Tables
#----------------------------------------------------------------------

# Import datasets
df_ords = pd.read_csv(PREP_DIR / 'orders_checked.csv', index_col=False)
df_prods = pd.read_csv(PREP_DIR / 'products_checked.csv', index_col=False)
df_dep = pd.read_csv(PREP_DIR / 'departments_wrangled.csv', index_col=False)
df_ords_prior = pd.read_csv(RAW_DIR / 'orders_products_prior.csv', index_col=False)

print(df_ords_prior.head())

print(df_ords.head())

print(df_ords_prior.shape)

print(df_ords.shape)

df_merged_large = df_ords.merge(df_ords_prior, on = 'order_id', indicator = True)

# check output
print(df_merged_large.head())

print(df_merged_large.shape)

print(df_merged_large['_merge'].value_counts())

# drop indicator flag (_merge)
df_merged_large = df_merged_large.drop(columns = ['_merge'])

print(df_merged_large.head())

df_merged_large.to_pickle(PREP_DIR / 'orders_products_combined.pkl')


#----------------------------------------------------------------------
# Merge Validation & Export
#----------------------------------------------------------------------

# Import datasets
df_ords = pd.read_csv(PREP_DIR / 'orders_checked.csv', index_col=False)
df_prods = pd.read_csv(PREP_DIR / 'products_checked.csv', index_col=False)
df_dep = pd.read_csv(PREP_DIR / 'departments_wrangled.csv', index_col=False)
df_ords_prior = pd.read_csv(RAW_DIR / 'orders_products_prior.csv', index_col=False)
df_ords_prods_comb = pd.read_pickle(PREP_DIR / 'orders_products_combined.pkl')

print(df_ords_prods_comb.shape)

print(df_ords_prods_comb.head())

print(df_prods.head())

print(df_prods.shape)

print(df_prods[df_prods.duplicated()])  # full-row duplicates

df_dups = df_prods[df_prods.duplicated(subset=['product_id'])]
print(df_dups)

df_prods = df_prods.drop_duplicates(subset=['product_id'])

print(df_prods.shape)

df_merged = df_ords_prods_comb.merge(df_prods, on = 'product_id', how = 'left', indicator = True)

print(df_merged['_merge'].value_counts())

# expect 32434489 rows
print(df_merged.shape)

df_merged = df_merged.drop(columns = ['_merge'])

print(df_merged.head())

df_merged.to_pickle(PREP_DIR / 'ords_prods_merge.pkl')
