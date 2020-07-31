# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 09:25:02 2020

@author: Fabretto

Script to classify Datazon UK retail customers in one of the 7 classes
identified in the exploratory data analysis.

Input:
    customer sequence: csv file in the current folder with invoice lines
        Data columns (total 9 columns):
        InvoiceNo      non-null int64
        StockCode      non-null object
        Description    non-null object
        Quantity       non-null int64
        InvoiceDate    non-null datetime64[ns]
        UnitPrice      non-null float64
        CustomerID     non-null float64
        Country        non-null object
    
    final models: joblib file with results from modelling notebook.
    
Notes:  1. Only UK data will be processed.
        2. Classes description as from the exploratory analysis results
"""

# Import libraries necessary for this project
import pandas as pd

from IPython.display import display # Allows the use of display() for DataFrames

from joblib import load

print('\nClassification of Datazon online retailers')
print('==========================================\n')

# Load Models
saved_models_path = 'final_models.joblib'

with open(saved_models_path,'rb') as f:
    reduced_pipeline = load(f)
    tuned_models = load(f)
    model_names = load(f)
    train_scores = load(f)
    train_miss_ndxes = load(f)
    elapsed_times = load(f)
    attributes = load(f)
    min_max_attr = load(f)
    power_attr = load(f)

# Description of the classes as from the exploratory analysis
class_desc = {
    0:
    'High sales frequency, low value, short product range and low week-end sales',
    1: 'Recent sales of large quantities of cheap items with high total value',
    2:
    'High sales frequencies with low value and some recent sales of cheap items with high total value',
    3: 'Sales with high total value',
    4: 'Week-end sales, low value',
    5: 'Mostly low total value',
    6: 'Week-end sales'}

# Load Customer Sequence
customer_seq_path = 'customer_seq.csv'

# The expected header:
# InvoiceNo	StockCode Description Quantity InvoiceDate UnitPrice CustomerID Country
# Parse InvoiceDate
customer_seq = pd.read_csv(customer_seq_path, parse_dates=[4])
# customer_seq.info()

# Ensure only UK data is taken into account
if len(customer_seq[customer_seq['Country'] != 'United Kingdom']) > 0:
    print('\nOnly sales in the United Kingdom are taken into account.')
    print('{:d} lines have been skipped'.format(
            len(customer_seq[customer_seq['Country'] != 'United Kingdom'])))
customer_seq = customer_seq.loc[customer_seq['Country'] == 'United Kingdom']

print(
    '{:d} invoice lines with {:d} invoices for {:d} customers have been read.'.
    format(
        len(customer_seq), customer_seq['InvoiceNo'].nunique(),
        customer_seq['CustomerID'].nunique()))

# Add a column with total sales
customer_seq.loc[:,'TotalPrice'] = \
    customer_seq['Quantity'] * customer_seq['UnitPrice']

# Flag the weekend sales by invoice line
customer_seq.loc[:,'weekend_sale'] = \
    customer_seq.InvoiceDate.map(lambda x: 1 if x.isoweekday()>=6 else 0)

# Count week-sales lines per customer
df_weekend_sale = customer_seq.groupby('CustomerID')['weekend_sale'].agg(
    [('count_we', 'sum'), ('count_all', 'count')])

# Ratio of invoice lines during week-ends per customer.
df_weekend_ratio = pd.DataFrame(df_weekend_sale.count_we /
                                df_weekend_sale.count_all,
                                columns=['pct_invoice_line'])

# Monetary value per customer
cust_table = customer_seq.groupby('CustomerID')['TotalPrice'].agg(
    [('monetary_value', 'sum')])

# Number of distinct items ordered per customer. Customer product range.
df_customer_ordered_items = \
    customer_seq.groupby('CustomerID')['StockCode']\
    .agg([('nb_dist_articles','nunique')])

# Average, min and max ordered quantity per item
df_q_per_item = customer_seq.groupby('CustomerID')['Quantity'].agg(
    [('avg_q_item', 'mean'), ('min_q_item', 'min'), ('max_q_item', 'max')])

# Average, min and max ordered total price per item
df_tprice_per_item = customer_seq.groupby('CustomerID')['TotalPrice'].agg(
    [('avg_tprice_item', 'mean'),
     ('min_tprice_item', 'min'),
     ('max_tprice_item', 'max')])

# Average, min and max ordered unit price per item.
df_uprice_per_item = customer_seq.groupby('CustomerID')['UnitPrice'].agg(
    [('avg_uprice_item', 'mean'),
     ('min_uprice_item', 'min'),
     ('max_uprice_item', 'max')])

customers = cust_table.join([
    df_customer_ordered_items,
    df_q_per_item,
    df_tprice_per_item,
    df_uprice_per_item,
    df_weekend_ratio])

# Build predictions
final_model = \
    dict(zip(model_names, tuned_models)).get('SVC Linear').best_estimator_

X_prepared = reduced_pipeline.transform(customers)

predictions = final_model.predict(X_prepared)

print('\nClassification results')
print('----------------------\n')

for cust, cl in zip(customers.index, predictions):
    print('Customer {:.0f}: class Cl{}: {}'.format(cust, cl, class_desc[cl]))
