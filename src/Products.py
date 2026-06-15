"""Products module for reading and writing product data.

Handles product data from CSV files with special character handling and
data type conversions.
"""

import os
from pathlib import Path
from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.sql import SparkSession


class Products:

    def __init__(self, spark, data_dir):
        """Initialize Products class with Spark session and data directory path."""
        self.spark = spark
        self.data_dir = data_dir


    def read_products_csv(self):
        """Read product data from CSV file with special character handling.
        
        Returns:
            DataFrame with product data, or empty DataFrame if error occurs
        """
        try:
            schema = T.StructType([
                T.StructField("Product ID", T.StringType(), True),
                T.StructField("Category", T.StringType(), True),
                T.StructField("Sub-Category", T.StringType(), True),
                T.StructField("Product Name", T.StringType(), True),
                T.StructField("State", T.StringType(), True),
                T.StructField("Price Per Product", T.StringType(), True),
            ])
            return (
                self.spark.read.option("multiline", "true")
                .option("quote", '"').option("escape", '"')  # Handle quoted fields with escaped quotes
                .csv(os.path.join(self.data_dir, "files", "Products.csv"), header=True, schema=schema)
            )
        except Exception as e:
            print(f"Error reading products CSV: {e}")
            return self.spark.createDataFrame([], schema=schema)  # Return an empty DataFrame with the defined schema


    def write_products_raw(self, raw_df):
        """Write raw product DataFrame to parquet format.
        
        Args:
            raw_df: Raw product DataFrame with original column names
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save to raw directory with parquet format for efficient storage
            raw_df.write.mode("overwrite").parquet(os.path.join(self.data_dir, "raw", "products.parquet"))
            return True
        except Exception as e:
            print(f"Error writing products raw: {e}")
            return False
    

    def write_products_enriched(self, raw_df):
        """Transform and enrich product data with standardized column names and data types.
        
        Args:
            raw_df: Raw product DataFrame
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Normalize column names (CamelCase/PascalCase -> snake_case)
            # Cast price_per_product to double for numeric operations
            enriched_df = (
                raw_df
                .select(
                    F.col("Product ID").alias("product_id"),
                    F.col("Category").alias("category"),
                    F.col("Sub-Category").alias("sub_category"),
                    F.col("Product Name").alias("product_name"),
                    F.col("State").alias("state"),
                    F.col("Price Per Product").cast(T.DoubleType()).alias("price_per_product")  # Convert to double
                )
            )
            # Save enriched data to parquet format
            enriched_df.write.mode("overwrite").parquet(os.path.join(self.data_dir, "enriched", "products.parquet"))
            return True
        except Exception as e:
            print(f"Error writing products enriched: {e}")
            return False
        

    def read_products_enriched(self):
        """Read enriched product data from parquet file.
        
        Returns:
            DataFrame with enriched product data, or empty DataFrame if error occurs
        """
        try:
            # Define schema to match the enriched product table structure
            enriched_schema = T.StructType([
                T.StructField("product_id", T.StringType(), True),
                T.StructField("category", T.StringType(), True),
                T.StructField("sub_category", T.StringType(), True),
                T.StructField("product_name", T.StringType(), True),
                T.StructField("state", T.StringType(), True),
                T.StructField("price_per_product", T.DoubleType(), True),
            ])
            return self.spark.read.parquet(os.path.join(self.data_dir, "enriched", "products.parquet"), schema=enriched_schema)
        except Exception as e:
            print(f"Error reading products enriched: {e}")
            return self.spark.createDataFrame([], schema=enriched_schema)  # Return an empty DataFrame with the defined schema