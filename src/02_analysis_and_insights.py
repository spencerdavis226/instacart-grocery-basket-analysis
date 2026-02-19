"""
02 — Analysis & Insights (Instacart)

Generated from course notebooks via tools/merge_course_notebooks.py
"""


#----------------------------------------------------------------------
# Feature Engineering
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

ANALYSIS_DIR = BASE_DIR / '04 Analysis'
VIZ_DIR = ANALYSIS_DIR / 'Visualizations'

# Make sure output folders exist
PREP_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Import datasets
ords_prods_merge = pd.read_pickle(PREP_DIR / 'ords_prods_merge.pkl')

# Create subset with 1,000,000 rows
df = ords_prods_merge[:1000000]

# View data for reference
print(df.head())

print(df.shape)

def price_label(row):

  if row['prices'] <= 5:
    return 'Low-range product'
  elif (row['prices'] > 5) and (row['prices'] <= 15):
    return 'Mid-range product'
  elif row['prices'] > 15:
    return 'High range'
  else: return 'Not enough data'

# Define new column 'price_range' based on condition. Axis=1 stands for "rows" (0 = apply to columns)
df['price_range'] = df.apply(price_label, axis=1)

print(df['price_range'].value_counts(dropna=False))

print(df['prices'].max())

df.loc[df['prices'] > 15, 'price_range_loc'] = 'High-range product'

df.loc[(df['prices'] <= 15) & (df['prices'] > 5), 'price_range_loc'] = 'Mid-range product'

df.loc[df['prices'] <= 5, 'price_range_loc'] = 'Low-range product'

print(df['price_range_loc'].value_counts(dropna=False))

ords_prods_merge.loc[ords_prods_merge['prices'] > 15, 'price_label'] = 'High-range product'

ords_prods_merge.loc[(ords_prods_merge['prices'] <= 15) & (ords_prods_merge['prices'] > 5), 'price_label'] = 'Mid-range product'

ords_prods_merge.loc[ords_prods_merge['prices'] <= 5, 'price_label'] = 'Low-range product'

print(ords_prods_merge['price_label'].value_counts(dropna=False))

# View busiest days of week. 0 = Saturday
print(ords_prods_merge['orders_day_of_week'].value_counts(dropna=False))

# Create empty list
result = []

# Loop through dataframe
for value in ords_prods_merge["orders_day_of_week"]:
  if value == 0:
    result.append("Busiest day")
  elif value == 4:
    result.append("Least busy")
  else:
    result.append("Regularly busy")

# Make new column 'busiest day' and combine with result
ords_prods_merge['busiest_day'] = result

print(ords_prods_merge['busiest_day'].value_counts(dropna=False))

# Verify I have “price_label” and “busiest_day” columns.
print(ords_prods_merge.head())

# View busiest days of week. 0 = Saturday
print(ords_prods_merge['orders_day_of_week'].value_counts(dropna=False))

# Create busiest days column

# Create empty list
result2 = []

# Loop through dataframe, using OR logic
for value in ords_prods_merge["orders_day_of_week"]:
  if value == 0 or value == 1:
    result2.append("Busiest days")
  elif value == 4 or value == 3:
    result2.append("Least busy")
  else:
    result2.append("Regularly busy")

# Make new column 'busiest days' and combine with result
ords_prods_merge['busiest_days'] = result2

# View distribution
print(ords_prods_merge['busiest_days'].value_counts(dropna=False))

# Expect 32,434,489 rows
print(ords_prods_merge.shape)

# View spread of data
print(ords_prods_merge['order_hour_of_day'].value_counts(dropna=False))

# Define lists of hours, split into thirds (8 hours each group)
most_orders = [10, 11, 14, 15, 13, 12, 16, 9]
average_orders = [17, 8, 18, 19, 20, 7, 21, 22]
fewest_orders = [23, 6, 0, 1, 5, 2, 4, 3]

# Apply labels using .loc since it is cleaner and faster
ords_prods_merge.loc[ords_prods_merge['order_hour_of_day'].isin(most_orders), 'busiest_period_of_day'] = 'Most orders'
ords_prods_merge.loc[ords_prods_merge['order_hour_of_day'].isin(average_orders), 'busiest_period_of_day'] = 'Average orders'
ords_prods_merge.loc[ords_prods_merge['order_hour_of_day'].isin(fewest_orders), 'busiest_period_of_day'] = 'Fewest orders'

# Print frequency
print(ords_prods_merge['busiest_period_of_day'].value_counts())

ords_prods_merge.to_pickle(PREP_DIR / 'orders_products_merged_updated.pkl')


#----------------------------------------------------------------------
# Aggregations
#----------------------------------------------------------------------

# Import datasets
ords_prods_merge = pd.read_pickle(PREP_DIR / 'orders_products_merged_updated.pkl')

# Create subset of first 1M
df = ords_prods_merge[:1000000]

print(df.shape)

print(df.head())

df.groupby('product_name')

df.groupby('department_id').agg({'order_number': ['mean']})

# Same result without agg() function
df.groupby('department_id')['order_number'].mean()

df.groupby('department_id').agg({'order_number': ['mean', 'min', 'max']})

# All 3 steps in 1 code
ords_prods_merge['max_order'] = ords_prods_merge.groupby(['user_id'])['order_number'].transform("max")

print(ords_prods_merge.head(100))

# Create loyalty flags based on max orders
ords_prods_merge.loc[ords_prods_merge['max_order'] > 40, 'loyalty_flag'] = 'Loyal customer'
ords_prods_merge.loc[(ords_prods_merge['max_order'] <= 40) & (ords_prods_merge['max_order'] > 10), 'loyalty_flag'] = 'Regular customer'
ords_prods_merge.loc[ords_prods_merge['max_order'] <= 10, 'loyalty_flag'] = 'New customer'

# Check values
print(ords_prods_merge['loyalty_flag'].value_counts())

# Check head() of only columns of interest using df['column']
print(ords_prods_merge[['user_id', 'loyalty_flag', 'order_number']].head(60))

# Task 1 verification
print(ords_prods_merge.head())

# 2. Create aggregated mean of “order_number” column grouped by “department_id”
ords_prods_merge.groupby('department_id')['order_number'].mean()

# 3. Analyze the result
# Compare to subset results:
df.groupby('department_id')['order_number'].mean()

# 5.The marketing team at Instacart wants to know whether there’s a difference between the spending habits of the three types of customers you identified.
ords_prods_merge.groupby('loyalty_flag')['prices'].mean()

# Outlier inspection
print(ords_prods_merge['prices'].describe())

print(ords_prods_merge.loc[ords_prods_merge['prices'] > 100])

print(ords_prods_merge.loc[ords_prods_merge['prices'] > 90000])

# Turn values > 100 into NaNs
ords_prods_merge.loc[ords_prods_merge['prices'] > 100, 'prices'] = np.nan

# Check for outliers again
print(ords_prods_merge.loc[ords_prods_merge['prices'] > 50])

# Complete task 5 again with cleaned data
ords_prods_merge.groupby('loyalty_flag')['prices'].mean()

# 6. Create a spending flag for each user based on the average price across all their orders
# Create spending flag for each user
ords_prods_merge['user_avg_item_price'] = ords_prods_merge.groupby(['user_id'])['prices'].transform("mean")

print(ords_prods_merge.head(100))

# Create spending flags based on avg order price
ords_prods_merge.loc[ords_prods_merge['user_avg_item_price'] < 10, 'spending_habit'] = 'Low spender'
ords_prods_merge.loc[ords_prods_merge['user_avg_item_price'] >= 10, 'spending_habit'] = 'High spender'

# Check values
print(ords_prods_merge['spending_habit'].value_counts())

# 7. Create an order frequency flag that marks the regularity of a user’s ordering behavior
# Median days between orders
ords_prods_merge['median_days_between_orders'] = ords_prods_merge.groupby(['user_id'])['days_since_prior_order'].transform("median")

print(ords_prods_merge.head(100))

# Create user frequency flags based on median days since prior order
ords_prods_merge.loc[ords_prods_merge['median_days_between_orders'] > 20, 'order_frequency_flag'] = 'Non-frequent customer'
ords_prods_merge.loc[(ords_prods_merge['median_days_between_orders'] <= 20) & (ords_prods_merge['median_days_between_orders'] > 10), 'order_frequency_flag'] = 'Regular customer'
ords_prods_merge.loc[ords_prods_merge['median_days_between_orders'] <= 10, 'order_frequency_flag'] = 'Frequent customer'

# Check values
print(ords_prods_merge['order_frequency_flag'].value_counts())

print(ords_prods_merge.head())

# 9. Export your dataframe as a pickle file and store it correctly in your “Prepared Data” folder.
ords_prods_merge.to_pickle(PREP_DIR / 'orders_products_aggregated.pkl')


#----------------------------------------------------------------------
# Visualizations & Demographics
#----------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
import scipy

# Import datasets
ords_prods_cust_merge = pd.read_pickle(PREP_DIR / 'orders_products_all.pkl')

# Create histogram for order_hour_of_day
hours = range(24)

ax = (ords_prods_cust_merge["order_hour_of_day"]
      .value_counts()
      .reindex(hours, fill_value=0)
      .plot(kind="bar"))

ax.set_title("Order Frequency by Hour of Day")
ax.set_xlabel("Hour of Day (0–23)")
ax.set_ylabel("Number of Orders")
plt.tight_layout()
plt.show()

# Create bar chart for loyalty_flag
bar = ords_prods_cust_merge['loyalty_flag'].value_counts().plot.bar()

# Create the seed (ensures reproducibility)
np.random.seed(4)

# Create a list holding True/False values to the test np.random.rand() <= 0.7
dev = np.random.rand(len(ords_prods_cust_merge)) <= 0.7

# Store 70% of the sample in the dataframe 'big'
big = ords_prods_cust_merge[dev]

# Store 30% of the sample in the dataframe 'small'
small = ords_prods_cust_merge[~dev]

# Verify row counts
len(ords_prods_cust_merge)

len(big) + len(small)

# Create the line chart using the small sample
line = sns.lineplot(data = small, x = 'order_hour_of_day', y = 'prices')

# Create line chart for Age vs. Number of Dependents
line_2 = sns.lineplot(data = small, x = 'age', y = 'number_of_dependents')

# Create scatterplot for Age vs. Income
scatter = sns.scatterplot(x = 'age', y = 'income', data = ords_prods_cust_merge)

ax.figure.savefig(VIZ_DIR / 'hist_order_hour_of_day.png')

bar.figure.savefig(VIZ_DIR / 'bar_loyalty_flag.png')

line.figure.savefig(VIZ_DIR / 'line_prices_hour_of_day.png')

line_2.figure.savefig(VIZ_DIR / 'line_age_dependents.png')

scatter.figure.savefig(VIZ_DIR / 'scatter_age_income.png')


#----------------------------------------------------------------------
# Regional Segmentation
#----------------------------------------------------------------------

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

df = pd.read_pickle(PREP_DIR / 'orders_products_all.pkl')
df_dept = pd.read_csv(RAW_DIR / 'departments.csv', index_col=False)

print(df.columns)

# Drop PII columns
df = df.drop(columns=['first_name', 'surname'])

# Check to ensure they are gone
print(df.columns)

# View States
print(df['state'].value_counts())

region_northeast = ['Maine', 'New Hampshire', 'Vermont', 'Massachusetts', 'Rhode Island', 'Connecticut', 'New York', 'Pennsylvania', 'New Jersey']

region_midwest = ['Wisconsin', 'Michigan', 'Illinois', 'Indiana', 'Ohio', 'North Dakota', 'South Dakota', 'Nebraska', 'Kansas', 'Minnesota', 'Iowa', 'Missouri']

region_south = ['Delaware', 'Maryland', 'District of Columbia', 'Virginia', 'West Virginia', 'North Carolina', 'South Carolina', 'Georgia', 'Florida', 'Kentucky', 'Tennessee', 'Mississippi', 'Alabama', 'Oklahoma', 'Texas', 'Arkansas', 'Louisiana']

region_west = ['Idaho', 'Montana', 'Wyoming', 'Nevada', 'Utah', 'Colorado', 'Arizona', 'New Mexico', 'Alaska', 'Washington', 'Oregon', 'California', 'Hawaii']

df.loc[df['state'].isin(region_northeast), 'region'] = 'Northeast'

df.loc[df['state'].isin(region_midwest), 'region'] = 'Midwest'

df.loc[df['state'].isin(region_south), 'region'] = 'South'

df.loc[df['state'].isin(region_west), 'region'] = 'West'

# Check the output
df['region'].value_counts(dropna=False)

# Create a crosstab
crosstab = pd.crosstab(df['region'], df['spending_habit'], dropna = False)

print(crosstab)

# Create a bar chart
bar_chart = crosstab.plot.bar(color=['lightblue', 'darkblue'])

# Export chart
bar_chart.figure.savefig(VIZ_DIR / 'bar_regional_spending_habits.png')

# Create the activity flag
df.loc[df['max_order'] < 5, 'activity_flag'] = 'Low activity'
df.loc[df['max_order'] >= 5, 'activity_flag'] = 'High activity'

# Check the count of each flag
df['activity_flag'].value_counts(dropna = False)

# Create a subset of ONLY high activity customers
df_active = df[df['activity_flag'] == 'High activity']

# Check the shape to verify rows were dropped
print(df_active.shape)

# Export the active customers data
df_active.to_pickle(PREP_DIR / 'orders_products_active.pkl')

# Shift dataframe to only high activity customers
df = df_active

# Create age_group column
df.loc[df['age'] <= 35, 'age_group'] = 'Young Adult'
df.loc[(df['age'] > 35) & (df['age'] < 65), 'age_group'] = 'Middle Aged'
df.loc[df['age'] >= 65, 'age_group'] = 'Senior'

# Check the distribution
df['age_group'].value_counts(dropna=False)

# Create income_group column
df.loc[df['income'] < 50000, 'income_group'] = 'Low Income'
df.loc[(df['income'] >= 50000) & (df['income'] <= 120000), 'income_group'] = 'Middle Income'
df.loc[df['income'] > 120000, 'income_group'] = 'High Income'

# Check the distribution
df['income_group'].value_counts(dropna=False)

# Create dependent_flag
df.loc[df['number_of_dependents'] > 0, 'dependent_flag'] = 'Parent'
df.loc[df['number_of_dependents'] == 0, 'dependent_flag'] = 'Non-parent'

# Check the distribution
df['dependent_flag'].value_counts(dropna=False)

# Young Parent
df.loc[(df['age_group'] == 'Young Adult') & (df['dependent_flag'] == 'Parent'), 'customer_profile'] = 'Young Parent'

# Single Adult
df.loc[(df['age_group'] == 'Young Adult') & (df['dependent_flag'] == 'Non-parent'), 'customer_profile'] = 'Single Adult'

# Middle Aged Parent
df.loc[(df['age_group'] == 'Middle Aged') & (df['dependent_flag'] == 'Parent'), 'customer_profile'] = 'Middle Aged Parent'

# Single Senior
df.loc[(df['age_group'] == 'Senior') & (df['dependent_flag'] == 'Non-parent'), 'customer_profile'] = 'Single Senior'

# Senior Parent
df.loc[(df['age_group'] == 'Senior') & (df['dependent_flag'] == 'Parent'), 'customer_profile'] = 'Senior Parent'

# Single Middle Aged
df.loc[(df['age_group'] == 'Middle Aged') & (df['dependent_flag'] == 'Non-parent'), 'customer_profile'] = 'Single Middle Aged'

# Check the distribution
df['customer_profile'].value_counts(dropna=False)

# Create customer profile summary
table_profile_summary = df.groupby('customer_profile').agg({'max_order': ['mean', 'min', 'max'], 'prices': ['mean', 'min', 'max'], 'days_since_prior_order': ['mean']}).round(2)

table_profile_summary

# Create a flag for users who have ordered from the baby department (ID 18)
df['baby_department'] = [1 if x == 18 else 0 for x in df['department_id']]

# Assign flag if user has ever bought from dept 18
df['has_baby_status'] = df.groupby(['user_id'])['baby_department'].transform('max')

# Label the profile for clarity
df.loc[df['has_baby_status'] == 1, 'baby_status'] = 'Baby Household'
df.loc[df['has_baby_status'] == 0, 'baby_status'] = 'Non-baby Household'

# Drop the temp column
df = df.drop(columns=['baby_department'])

# Check the distribution
df['baby_status'].value_counts(dropna=False)

# Create a bar chart for Customer Profile
bar_profile = df['customer_profile'].value_counts().plot.bar(title='Distribution of Customer Profiles')

# Save the visualization
bar_profile.figure.savefig(os.path.join(path, '04 Analysis','Visualizations', 'bar_customer_profile.png'))

# Group by Profile AND Income to see the spending differences
df.groupby(['customer_profile', 'income_group']).agg({'prices': ['mean', 'min', 'max'], 'days_since_prior_order': ['mean', 'min', 'max']})

# Aggregation for baby status
df.groupby(['baby_status', 'income_group']).agg({'prices': ['mean', 'max', 'min'], 'days_since_prior_order': ['mean', 'max', 'min']})

# Create a crosstab to compare Customer Profiles and Regions
crosstab_region = pd.crosstab(df['customer_profile'], df['region'], dropna=False)

# Check the table
print(crosstab_region)

# Optional: enforce a consistent region order (adjust if yours differ)
region_order = ["Midwest", "Northeast", "South", "West"]
cols = [c for c in region_order if c in crosstab_region.columns] or list(crosstab_region.columns)

# Sort profiles by total orders across regions (descending)
ct = (crosstab_region[cols]
      .assign(_total=crosstab_region[cols].sum(axis=1))
      .sort_values("_total", ascending=False)
      .drop(columns="_total"))


bar_region_profile = ct.plot(kind="bar")
ax = bar_region_profile

ax.set_title("Customer Profile by Region")
ax.set_xlabel("Customer Profile")
ax.set_ylabel("Number of Orders")
ax.legend(title="Region", frameon=False)

plt.xticks(rotation=25, ha="right")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1_000_000:.1f}M"))

plt.tight_layout()
plt.show()

# Export the chart
bar_region_profile.figure.savefig(VIZ_DIR / 'bar_profile_region.png')

# Transpose the departments dataframe
df_dept_t = df_dept.T
df_dept_t.reset_index()

# Create a new header
new_header = df_dept_t.iloc[0]
df_dept_t = df_dept_t[1:]
df_dept_t.columns = new_header

# Check the output to ensure it looks right
print(df_dept_t)

# Create dictionary
data_dict = df_dept_t.to_dict('index')

# Flatten result (instead of {1: {'department': 'frozen'}}, we want {1: 'frozen'})
department_dict = {int(key): value['department'] for key, value in data_dict.items()}

# Map the dictionary to main dataframe
df['department'] = df['department_id'].map(department_dict)

# Check the result
print(df[['department_id', 'department']].head())

# Verify the count
df['department'].value_counts(dropna=False)

# Fill NaNs with the string 'missing'
df['department'] = df['department'].fillna('missing')

# Verify the count
df['department'].value_counts(dropna=False)

# Create a crosstab to compare Customer Profiles and Departments
crosstab_dept = pd.crosstab(df['department'], df['customer_profile'], dropna=False)

print(crosstab_dept)

# Create a stacked bar chart
bar_dept_profile = crosstab_dept.plot.bar(stacked=True)
plt.title('Department Orders by Customer Profile')
plt.ylabel('Frequency')

# Export stacked bar chart
bar_dept_profile.figure.savefig(VIZ_DIR / 'bar_dept_profile.png')

print(df.shape)

# Create loyalty behavior summary table
table_loyalty_summary = df.groupby('loyalty_flag').agg({
    'max_order': ['mean', 'min', 'max'],
    'prices': ['mean', 'min', 'max'],
    'days_since_prior_order': ['mean']
}).round(2)

print(table_loyalty_summary)

# Make price range vs price_label table
price_counts = df['price_label'].value_counts(dropna=False)

price_pct = (df['price_label'].value_counts(normalize=True, dropna=False) * 100).round(1)

price_label_mix = pd.concat([price_counts, price_pct], axis=1)

price_label_mix.columns = ['order_lines', 'percent']

print(price_label_mix)

# Make revenue by hour (real order totals + AOV)

# Build order-level totals (sum of prices per order_id)
df_orders = (
    df.groupby("order_id", as_index=False)
      .agg(
          order_hour_of_day=("order_hour_of_day", "first"),
          order_total=("prices", "sum")
      )
)

# Summarize by hour
hourly_rev = (
    df_orders.groupby("order_hour_of_day", as_index=False)
             .agg(
                 total_revenue=("order_total", "sum"),
                 avg_order_value=("order_total", "mean"),
                 median_order_value=("order_total", "median"),
                 n_orders=("order_total", "size")
             )
             .sort_values("order_hour_of_day")
)

print(hourly_rev)

hourly_rev.plot(x="order_hour_of_day", y="total_revenue", kind="bar", figsize=(10,4), legend=False, title="Total revenue by hour")

hourly_rev.plot(x="order_hour_of_day", y="avg_order_value", kind="line", figsize=(10,4), legend=False, title="Average order value (AOV) by hour")

# Export the final data set with the fixed 'department' column
df.to_pickle(PREP_DIR / 'orders_products_final_profiles.pkl')
