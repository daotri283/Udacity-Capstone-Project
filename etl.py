#Importing the neccessary library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg
from pyspark.sql import SQLContext
from pyspark.sql.functions import isnan, when, trim, count, upper, col, udf, dayofmonth, dayofweek, month, year, weekofyear
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import *
from datetime import timedelta, datetime
import configparser
import psycopg2

from util import *
from sql_querries import *

config = configparser.ConfigParser()
config.read('config.cfg')

def create_spark_session():
    spark = SparkSession.builder\
                        .config("spark.jars.repositories", "https://repos.spark-packages.org/")\
                        .config("spark.jars.packages", "saurfang:spark-sas7bdat:2.0.0-s_2.11")\
                        .enableHiveSupport().getOrCreate()
    return spark

def create_immigration_fact_table(spark, input_path,output_path):
    """This function creates an country dimension from the immigration  data.
    :param spark: spark session
    :param input_path: path of immigration_data
    :param output_path: path to write Immigration Fact dataframe to
    :return: spark dataframe representing Immigration Fact data
    """
    immigration_df = spark.read.format('com.github.saurfang.sas.spark').load(input_path)
    immigration_df.write.parquet("sas_data")
    immigration_df = spark.read.parquet("sas_data")
    drop_col_list = ['occup', 'entdepu', 'insnum']
    immigration_df = drop_null_column(immigration_df, drop_col_list)
    immigration_df = drop_duplicate_row(immigration_df, subset=['cicid'])
    immigration_fact_table = immigration_df.withColumnRenamed('cicid', 'record_id') \
                                            .withColumnRenamed('i94res', 'country_residence_code') \
                                            .withColumnRenamed('i94addr', 'state_code')

    immigration_fact_table = immigration_fact_table.withColumn("country_residence_code",immigration_fact_table.country_residence_code.cast(IntegerType()))

    return immigration_fact_table


def create_time_dimension_table(spark, immigration_data_input_path,output_path):
    """This function creates an immigration calendar based on arrdate column from immigration fact table
    :param spark: spark session
    :param immigration_data_input_path: path of immigration_data
    :param output_path: S3 path to write dimension dataframe to
    :return: spark dataframe representing Time Dimension data
    """
    immigration_df = spark.read.format('com.github.saurfang.sas.spark').load(immigration_data_input_path)
    # immigration_df.write.parquet("sas_data")
    immigration_df = spark.read.parquet("sas_data")
    drop_col_list = ['occup', 'entdepu', 'insnum']
    immigration_df = drop_null_column(immigration_df, drop_col_list)
    immigration_df = drop_duplicate_row(immigration_df, subset=['cicid'])
    immigration_df = immigration_df.withColumn("arrdate", convert_timestamp(immigration_df.arrdate))

    time_dim_table = immigration_df.select(['arrdate'])
    time_dim_table = time_dim_table.withColumn('arr_day', dayofmonth('arrdate'))
    time_dim_table = time_dim_table.withColumn('arr_week', weekofyear('arrdate'))
    time_dim_table = time_dim_table.withColumn('arr_month', month('arrdate'))
    time_dim_table = time_dim_table.withColumn('arr_year', year('arrdate'))
    time_dim_table = time_dim_table.withColumn('arr_weekday', dayofweek('arrdate'))

    # create an id field for time dimension table
    time_dim_table = time_dim_table.withColumn('id', monotonically_increasing_id())

    return time_dim_table

def create_demographics_dimension_table(spark, input_path, output_path):
    """This function creates a us demographics dimension table from the us cities demographics data.
    :param spark: spark session
    :param input_path: demographics data path
    :param output_path: S3 path to write dimension dataframe to
    :return: spark dataframe representing demographics dimension table
    """
    demographics_df = spark.read.csv(input_path, inferSchema=True, header=True, sep=';')
    demographics_df = drop_duplicate_row(demographics_df)

    demographic_dim_table = demographics_df.withColumnRenamed('Median Age', 'median_age') \
                                        .withColumnRenamed('Male Population', 'male_population') \
                                        .withColumnRenamed('Female Population', 'female_population') \
                                        .withColumnRenamed('Total Population', 'total_population') \
                                        .withColumnRenamed('Number of Veterans', 'number_of_veterans') \
                                        .withColumnRenamed('Foreign-born', 'foreign_born') \
                                        .withColumnRenamed('Average Household Size', 'average_household_size') \
                                        .withColumnRenamed('State Code', 'state_code')
    # Add id column
    demographic_dim_table = demographic_dim_table.withColumn('id', monotonically_increasing_id())

    return demographic_dim_table


def create_temperature_dimension_table(spark, input_path, output_path):
    """This function creates a  temperature dimension table from the world temperature data.
    :param spark: spark session
    :param input_path: temperature data path
    :param output_path: S3 path to write temperature dataframe to
    :return: spark dataframe representing Temperature Dimension table
    """
    temperature_df = spark.read.csv(input_path, header=True, inferSchema=True)
    drop_col_list = ['AverageTemperature', 'AverageTemperatureUncertainty']
    temperature_df = temperature_df.dropna(how='any', subset=drop_col_list)
    temperature_df = drop_duplicate_row(temperature_df)
    temperature_df = temperature_df.select("*", upper(col('Country')))\
                                   .withColumnRenamed('upper(Country)', 'country_name')
    temperature_df = temperature_df

    with open('I94_SAS_Labels_Descriptions.SAS') as f:
        contents = f.readlines()

    country_list = []
    code_list = []
    for countries in contents[10:298]:
        pair = countries.split('=')
        code, country = pair[0].strip(), pair[1].strip().strip("'")
        country_list.append(country)
        code_list.append(code)
    country_code_df = {'code': code_list, 'country': country_list}
    country_code_df = pd.DataFrame(country_code_df)

    temperature_dim_table = temperature_df.select(['dt', 'AverageTemperature', 'AverageTemperatureUncertainty', \
                                                   'City', 'country_name']) \
        .distinct()

    temperature_dim_table = temperature_dim_table.select('*').groupby('country_name').avg()

    temperature_dim_table = temperature_dim_table.withColumnRenamed('avg(AverageTemperature)', 'average_temperature')
    temperature_dim_table = temperature_dim_table.withColumnRenamed('avg(AverageTemperatureUncertainty)',
                                                                    'average_temperature_uncertainty')

    @udf('string')
    def get_country_code(name):
        if name.strip() in list(country_code_df['country']):
            code = country_code_df[country_code_df['country'] == name.strip()]['code'].iloc[0]
            return code
        else:
            return '996'

    temperature_dim_table = temperature_dim_table.withColumn("country_code",
                                                             get_country_code(temperature_dim_table.country_name))
    # Create country code column
    temperature_dim_table = temperature_dim_table.withColumn("country_code",temperature_dim_table.country_code.cast(IntegerType()))

    temperature_dim_table = temperature_dim_table.withColumn('id', monotonically_increasing_id())
    return temperature_dim_table


def create_airport_code_dimension_table(spark, input_path, output_path):
    """This function creates a  temperature dimension table from the world temperature data.
    :param spark: spark session
    :param input_path: airport code data path
    :param output_path: S3 path to write airport codes dataframe to
    :return: spark dataframe representing Airport Code Dimension table
    """
    airport_code_df = spark.read.csv(input_path, inferSchema=True, header=True, sep=',')
    airport_code_df = airport_code_df.dropna(how='any', subset=['iata_code'])
    airport_code_df = drop_duplicate_row(airport_code_df)

    airport_code_dim_table = airport_code_df.select('*')
    # Create an id column
    airport_code_dim_table = airport_code_dim_table.withColumn('id', monotonically_increasing_id())

    return airport_code_dim_table

def insert_tables(cur, conn):
    """
    This function insert data from staging into fact and dimension tables in Redshift
    """
    for query in copy_table_queries:
        cur.execute(query)
        conn.commit()

# Press the green button in the gutter to run the script.
def main():
    # Create spark session
    spark = create_spark_session()

    # Get input path
    immigration_input_path = config['INPUT_PATH']['IMMIGRATION_INPUT_PATH']
    demographic_input_path = config['INPUT_PATH']['DEMOGRAPHICS_INPUT_PATH']
    airport_code_input_path = config['INPUT_PATH']['AIRPORT_CODE_INPUT_PATH']
    temperature_input_path = config['INPUT_PATH']['TEMPERATURE_INPUT_PATH']

    # Get output path
    output_path = config['S3']['S3_OUTPUT']

    immigration_fact_table = create_immigration_fact_table(spark, immigration_input_path, output_path)
    time_dimension_table = create_time_dimension_table(spark, immigration_input_path, output_path)
    demographics_dimension_table = create_demographics_dimension_table(spark, demographic_input_path, output_path)
    airport_code_dimension_table = create_airport_code_dimension_table(spark, airport_code_input_path, output_path)
    temperature_dimension_table = create_temperature_dimension_table(spark, temperature_input_path, output_path)

    # data quality check 
    immigration_fact_check = data_count_check (immigration_fact_table, 'Immigration Fact Table') & \
                         unique_key_check(immigration_fact_table, 'record_id', 'Immigration Fact Table')
    if immigration_fact_check: 
        immigration_fact_table.write.parquet(output_path + "immigration_fact", mode="overwrite")

    time_dimension_check = data_count_check (time_dimension_table, 'Time Dimension Table') & \
                           unique_key_check(time_dimension_table, 'id', 'Time Dimension Table')
    if time_dimension_check: 
        partition_columns = ['arr_year', 'arr_month', 'arr_week']
        time_dimension_table.write.parquet(output_path + "time_dimension", partitionBy=partition_columns, mode="overwrite")

    demographics_dimension_check = data_count_check (demographics_dimension_table, 'Demographics Dimension Table') &\
                                   unique_key_check(demographics_dimension_table, 'id', 'Demographics Dimension Table')

    if demographics_dimension_check:
        demographics_dimension_table.write.parquet(output_path + "demographics_dimension", mode="overwrite")

    airport_code_check = data_count_check (airport_code_dimension_table, 'Airport Code Dimension Table') & \
                         unique_key_check(airport_code_dimension_table, 'id', 'Airport Code Dimension Table')
    if airport_code_check: 
        airport_code_dimension_table.write.parquet(output_path + "airport_code_dimension", mode="overwrite")

    temperature_dimension_check = data_count_check (temperature_dimension_table, 'Temperature Dimension Table') & \
                                  unique_key_check(temperature_dimension_table, 'id', 'Temperature Dimension Table')

    if temperature_dimension_check:
        temperature_dimension_table.write.mode("overwrite").parquet(path=output_path + 'temperature_dimension')

    # Copy data to Redshift 
    conn = psycopg2.connect("host={} dbname={} user={} password={} port={}".format(*config['CLUSTER'].values()))
    cur = conn.cursor()
    insert_tables(cur, conn)
    conn.close()


if __name__ == '__main__':
    main()