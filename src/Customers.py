"""Customers module for reading and writing customer data.

Handles customer data from Excel files with transformations including
data type conversions and enrichment.
"""

import os
import pandas as pd
from pathlib import Path
from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

class Customers:

    def __init__(self, spark, data_dir):
        """Initialize Customers class with Spark session and data directory path."""
        self.spark = spark
        self.data_dir = data_dir


    def read_customers_excel(self):
        """Read customer data from Excel file and convert to Spark DataFrame.
        
        Returns:
            DataFrame with customer data, or empty DataFrame if error occurs
        """
        try:
            # Define schema for Customer Excel file before reading
            customers_schema_pandas = {
                "Customer ID": str,
                "Customer Name": str,
                "email": str,
                "phone": str,
                "address": str,
                "Segment": str,
                "Country": str,
                "City": str,
                "State": str,
                "Postal Code": str,
                "Region": str,
            }

            # Read Excel file with schema validation
            customers_pd = pd.read_excel(
                os.path.join(self.data_dir, "files", "Customer.xlsx"),
                dtype=customers_schema_pandas
            )

            customers_schema = T.StructType([
                T.StructField("Customer ID", T.StringType(), True),
                T.StructField("Customer Name", T.StringType(), True),
                T.StructField("email", T.StringType(), True),
                T.StructField("phone", T.StringType(), True),
                T.StructField("address", T.StringType(), True),
                T.StructField("Segment", T.StringType(), True),
                T.StructField("Country", T.StringType(), True),
                T.StructField("City", T.StringType(), True),
                T.StructField("State", T.StringType(), True),
                T.StructField("Postal Code", T.StringType(), True),
                T.StructField("Region", T.StringType(), True),
            ])
            return self.spark.createDataFrame(customers_pd, schema=customers_schema)
        except Exception as e:
            print(f"Error reading customers Excel: {e}")
            return self.spark.createDataFrame([], schema=customers_schema)  # Return an empty DataFrame with the defined schema
        

    def write_customers_raw(self, raw_df):
        """Write raw customer DataFrame to parquet format.
        
        Args:
            raw_df: Raw customer DataFrame with original column names
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save to raw directory with parquet format for efficient storage
            raw_df.write.mode("overwrite").parquet(os.path.join(self.data_dir, "raw", "customers.parquet"))
            return True
        except Exception as e:
            print(f"Error writing customers raw: {e}")
            return False
        
    
    def write_customers_enriched(self, raw_df):
        """Transform and enrich customer data with standardized column names and data types.
        
        Args:
            raw_df: Raw customer DataFrame
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Normalize column names (CamelCase/PascalCase -> snake_case)
            # Cast postal_code to integer for proper numeric handling
            enriched_df = (
                raw_df
                .select(
                    F.col("Customer ID").alias("customer_id"),
                    F.col("Customer Name").alias("customer_name"),
                    F.col("email").alias("email"),
                    F.col("phone").alias("phone"),
                    F.col("address").alias("address"),
                    F.col("Segment").alias("segment"),
                    F.col("Country").alias("country"),
                    F.col("City").alias("city"),
                    F.col("State").alias("state"),
                    F.col("Postal Code").cast(T.IntegerType()).alias("postal_code"),  # Convert to integer
                    F.col("Region").alias("region")
                )
            )
            # Save enriched data to parquet format
            enriched_df.write.mode("overwrite").parquet(os.path.join(self.data_dir, "enriched", "customers.parquet"))
            return True
        except Exception as e:
            print(f"Error writing customers enriched: {e}")
            return False
        

    def read_customers_enriched(self):
        """Read enriched customer data from parquet file.
        
        Returns:
            DataFrame with enriched customer data, or empty DataFrame if error occurs
        """
        try:
            # Define schema to match the enriched customer table structure
            enriched_schema = T.StructType([
                T.StructField("customer_id", T.StringType(), True),
                T.StructField("customer_name", T.StringType(), True),
                T.StructField("email", T.StringType(), True),
                T.StructField("phone", T.StringType(), True),
                T.StructField("address", T.StringType(), True),
                T.StructField("segment", T.StringType(), True),
                T.StructField("country", T.StringType(), True),
                T.StructField("city", T.StringType(), True),
                T.StructField("state", T.StringType(), True),
                T.StructField("postal_code", T.IntegerType(), True),
                T.StructField("region", T.StringType(), True),
            ])
            return self.spark.read.parquet(os.path.join(self.data_dir, "enriched", "customers.parquet"), schema=enriched_schema)
        except Exception as e:
            print(f"Error reading customers enriched: {e}")
            return self.spark.createDataFrame([], schema=enriched_schema)  # Return an empty DataFrame with the defined schema