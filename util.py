import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta, datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg
from pyspark.sql import SQLContext
from pyspark.sql.functions import isnan, when, trim, count, upper, col, udf, dayofmonth, dayofweek, month, year, weekofyear
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import *


def drop_null_column(df, column_list):
    return df.drop(*column_list)

def drop_duplicate_row(df):
    return df.dropDuplicates()


def plot_null_count_sparkdf(df):
    # Getting null data
    null_immigration_df = df.select(
        [count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).toPandas()
    null_immigration_df = null_immigration_df.T
    null_immigration_df.columns = ['null_counts']
    null_immigration_df['null_counts_percentage'] = null_immigration_df['null_counts'] / df.count() * 100
    null_immigration_df['columns'] = null_immigration_df.index

    # Plotting null data
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x='columns', y='null_counts_percentage', data=null_immigration_df)
    ax.set_ylim(0, 100)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.show()

def convert_timestamp(arrdate):
    return (datetime(1960,1,1) + timedelta(days=int(arrdate)))

def data_count_check(df, table_name):
    """Count checks on fact and dimension table to ensure completeness of data.
        :param df: spark dataframe to check counts on
        :param table_name: corresponding name of table
        """
    total_count = df.count()

    if total_count == 0:
        print(f"Data quality check failed for {table_name} with zero records!")
        return False
    else:
        print(f"Data quality check passed for {table_name} with {total_count:,} records.")
        return True


def unique_key_check(df, unique_key_name, table_name):
    dupes = df.groupBy(unique_key_name).count().filter("count > 1")
    if dupes.count() == 0:
        print(f"Data quality pass for {table_name} with no duplicate on unique key column!")
        return True
    else: 
        print(f"Data quality failed for {table_name} with duplicate on unique key column!")
        return False