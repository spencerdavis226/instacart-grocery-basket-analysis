# Instacart Grocery Basket Analysis (Python EDA + Customer Segmentation)

**Goal:** Analyze grocery ordering behavior to identify demand patterns and customer segments that can inform marketing strategy and operational planning.

## Key Insights (high-level)
- **Weekend demand is highest** (Sat/Sun), with midweek being slower → supports promo scheduling and staffing decisions.
- **Peak ordering window is late morning to afternoon** (strongest around 10–15) → helps time ads and capacity planning.
- **Customer behavior differs by segment and region** → supports targeted campaigns by customer profile and geography.

## Visual Highlights
### Order Frequency by Day of Week
![Order Frequency by Day of Week](assets/order_freq_dow.png)

### Order Frequency by Hour of Day
![Order Frequency by Hour of Day](assets/order_freq_hour_of_day.png)

### Customer Profile by Region
![Customer Profile by Region](assets/cust_profile_by_region.png)

## Deliverables
- **Executive Summary (PDF):** `report/Instacart Executive Summary.pdf`
- **Notebooks:** See the `notebooks/` folder for full analysis workflow

## Tools
Python, pandas, NumPy, matplotlib/seaborn, SciPy, Jupyter

## Data Note
Based on the public Instacart 2017 dataset (via Kaggle). Some customer attributes were course-provided/synthetic for learning purposes.
