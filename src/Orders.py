"""Orders module for reading and writing order data.

Handles order data from JSON files with schema validation, date transformations,
and profit calculations.
"""

import os
import pandas as pd
from pathlib import Path
from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

class Orders:

    def __init__(self, spark, data_dir):
        """Initialize Orders class with Spark session and data directory path."""
        self.spark = spark
        self.data_dir = data_dir


    def read_orders_json(self):
        """Read order data from JSON file with schema validation.
        
        Returns:
            DataFrame with order data, or empty DataFrame if error occurs
        """
        try:
            # Define schema for Orders JSON file
            orders_schema = T.StructType([
                T.StructField("Row ID", T.IntegerType(), True),
                T.StructField("Order ID", T.StringType(), True),
                T.StructField("Order Date", T.StringType(), True),
                T.StructField("Ship Date", T.StringType(), True),
                T.StructField("Ship Mode", T.StringType(), True),
                T.StructField("Customer ID", T.StringType(), True),
                T.StructField("Product ID", T.StringType(), True),
                T.StructField("Quantity", T.IntegerType(), True),
                T.StructField("Price", T.DoubleType(), True),
                T.StructField("Discount", T.DoubleType(), True),
                T.StructField("Profit", T.DoubleType(), True),
            ])
            json_path = os.path.join(self.data_dir, "files", "Orders.json")
            # Read JSON with multiLine option to handle records spanning multiple lines
            return self.spark.read.option("multiLine", "true").json(json_path, schema=orders_schema)
        except Exception as e:
            print(f"Error reading orders JSON: {e}")
            return self.spark.createDataFrame([], schema=orders_schema)  # Return an empty DataFrame with the defined schema
    

    def write_orders_raw(self, raw_df):
        """Write raw order DataFrame to parquet format.
        
        Args:
            raw_df: Raw order DataFrame with original column names
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save to raw directory with parquet format for efficient storage
            raw_df.write.mode("overwrite").parquet(os.path.join(self.data_dir, "raw", "orders.parquet"))
            return True
        except Exception as e:
            print(f"Error writing orders raw: {e}")
            return False
        

    def write_orders_enriched(self, raw_df):
        """Transform and enrich order data with standardized column names, date conversions, and data types.
        
        Args:
            raw_df: Raw order DataFrame
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Normalize column names and convert dates from d/M/yyyy format to DateType
            # Round profit to 2 decimal places for consistency
            enriched_df = (
                raw_df
                .select(
                    F.col("Row ID").alias("row_id"),
                    F.col("Order ID").alias("order_id"),
                    F.to_date(F.col("Order Date"), "d/M/yyyy").alias("order_date"),  # Convert string to date
                    F.to_date(F.col("Ship Date"), "d/M/yyyy").alias("ship_date"),    # Convert string to date
                    F.col("Ship Mode").alias("ship_mode"),
                    F.col("Customer ID").alias("customer_id"),
                    F.col("Product ID").alias("product_id"),
                    F.col("Quantity").cast(T.IntegerType()).alias("quantity"),
                    F.col("Price").cast(T.DoubleType()).alias("price"),
                    F.col("Discount").cast(T.DoubleType()).alias("discount"),
                    F.col("Profit").cast(T.DoubleType()).alias("profit")
                )
            )
            # Save enriched data to parquet format
            enriched_df.write.mode("overwrite").parquet(os.path.join(self.data_dir, "enriched", "orders.parquet"))
            return True
        except Exception as e:
            print(f"Error writing orders enriched: {e}")
            return False
        

    def read_orders_enriched(self):
        """Read enriched order data from parquet file.
        
        Returns:
            DataFrame with enriched order data, or empty DataFrame if error occurs
        """
        try:
            # Define schema to match the enriched order table structure
            enriched_schema = T.StructType([
                T.StructField("row_id", T.IntegerType(), True),
                T.StructField("order_id", T.StringType(), True),
                T.StructField("order_date", T.DateType(), True),
                T.StructField("ship_date", T.DateType(), True),
                T.StructField("ship_mode", T.StringType(), True),
                T.StructField("customer_id", T.StringType(), True),
                T.StructField("product_id", T.StringType(), True),
                T.StructField("quantity", T.IntegerType(), True),
                T.StructField("price", T.DoubleType(), True),
                T.StructField("discount", T.DoubleType(), True),
                T.StructField("profit", T.DoubleType(), True),
            ])
            return self.spark.read.parquet(os.path.join(self.data_dir, "enriched", "orders.parquet"), schema=enriched_schema)
        except Exception as e:
            print(f"Error reading orders enriched: {e}")
            return self.spark.createDataFrame([], schema=enriched_schema)  # Return an empty DataFrame with the defined schemas