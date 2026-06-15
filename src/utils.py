"""Utility functions for data aggregation and transformations.

Contains functions for creating fact tables and calculating profit metrics
by various dimensions (year, category, customer).
"""

import os
from pyspark.sql import functions as F
from pyspark.sql import types as T

def create_order_fact(customers_df, orders_df, products_df, data_dir):
        """
        Create a fact table by joining orders, customers, and products data.

        Args:
            customers_df: DataFrame containing customer information
            orders_df: DataFrame containing order information
            products_df: DataFrame containing product information
            data_dir: Base directory for saving the enriched fact table
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Join order data with customer data (left join to keep all orders)
            # Then join with product data to get category and subcategory info
            order_fact_df = (
                orders_df
                .join(customers_df, orders_df.customer_id == customers_df.customer_id, "left")
                .join(products_df, orders_df.product_id == products_df.product_id, "left")
                .select(
                    # Order information - transaction details
                    orders_df.order_id,
                    orders_df.order_date,
                    orders_df.ship_date,
                    orders_df.ship_mode,
                    orders_df.quantity,
                    orders_df.price,
                    orders_df.discount,
                    F.round(orders_df.profit, 2).alias("profit"),  # Round to 2 decimal places
                    # Customer information - who made the purchase
                    customers_df.customer_id,
                    customers_df.customer_name,
                    customers_df.country,
                    # Product information - what was purchased
                    products_df.product_id,
                    products_df.category,
                    products_df.sub_category
                )
            )
            # Save the combined fact table to parquet for efficient querying
            order_fact_df.write.mode("overwrite").parquet(os.path.join(data_dir, "enriched", "order_fact.parquet"))
            return True
        except Exception as e:
            print(f"Error creating order fact table: {e}")
            return False
        
    
def read_order_fact(spark, data_dir):
    """Read the pre-built order fact table from parquet storage.
    
    Args:
        spark: SparkSession object
        data_dir: Base directory where order_fact.parquet is stored
        
    Returns:
        DataFrame with order fact data, or empty DataFrame if error occurs
    """
    try:
        # Define schema matching the structure of order_fact table
        # Contains dimensions: order info, customer info, product info
        order_fact_schema = T.StructType([
            T.StructField("order_id", T.StringType(), True),
            T.StructField("order_date", T.DateType(), True),
            T.StructField("ship_date", T.DateType(), True),
            T.StructField("ship_mode", T.StringType(), True),
            T.StructField("quantity", T.IntegerType(), True),
            T.StructField("price", T.DoubleType(), True),
            T.StructField("discount", T.DoubleType(), True),
            T.StructField("profit", T.DoubleType(), True),
            T.StructField("customer_id", T.StringType(), True),
            T.StructField("customer_name", T.StringType(), True),
            T.StructField("country", T.StringType(), True),
            T.StructField("product_id", T.StringType(), True),
            T.StructField("category", T.StringType(), True),
            T.StructField("sub_category", T.StringType(), True)
        ])
        # Read parquet file with schema validation
        return spark.read.parquet(os.path.join(data_dir, "enriched", "order_fact.parquet"), schema=order_fact_schema)
    except Exception as e:
        print(f"Error reading order fact table: {e}")
        return spark.createDataFrame([], schema=order_fact_schema)  # Return empty DataFrame on error
        

def create_profit_info(order_fact_df, data_dir):
    """
    Create an aggregate table showing profit by Year, Product Category, 
    Product Sub Category, and Customer.
    
    Args:
        order_fact_df: The order fact table DataFrame containing order, customer, and product info
        data_dir: Base directory for saving the aggregated profit information
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Extract year from order_date and aggregate profit metrics
        # Group by multiple dimensions for detailed analysis
        profit_info_df = (
            order_fact_df
            .withColumn("year", F.year(F.col("order_date")))  # Extract year from date
            .groupBy(
                F.col("year"),
                F.col("category").alias("product_category"),
                F.col("sub_category").alias("product_sub_category"),
                F.col("customer_name").alias("customer")
            )
            .agg(
                F.round(F.sum(F.col("profit")), 2).alias("total_profit"),  # Sum profit rounded to 2 decimals
                F.count(F.col("order_id")).alias("order_count")  # Count total orders per group
            )
            .orderBy(
                F.col("year").desc(),
                F.col("product_category"),
                F.col("product_sub_category"),
                F.col("customer")
            )
        )
        
        # Save aggregated data to parquet for efficient querying and reporting
        profit_info_df.write.mode("overwrite").parquet(os.path.join(data_dir, "aggregated", "profit_info.parquet"))
        return True
    except Exception as e:
        print(f"Error creating profit info table: {e}")
        return False
    

def read_profit_info(spark, data_dir):
    try:
        profit_info_schema = T.StructType([
            T.StructField("year", T.IntegerType(), True),
            T.StructField("product_category", T.StringType(), True),
            T.StructField("product_sub_category", T.StringType(), True),
            T.StructField("customer", T.StringType(), True),
            T.StructField("total_profit", T.DoubleType(), True),
            T.StructField("order_count", T.LongType(), True)
        ])
        return spark.read.parquet(os.path.join(data_dir, "aggregated", "profit_info.parquet"), schema=profit_info_schema)
    except Exception as e:
        print(f"Error reading profit info table: {e}")
        return spark.createDataFrame([], schema=profit_info_schema)  # Return an empty DataFrame with the defined schema


def profit_by_year(spark, order_fact_df):
    """
    Calculate total profit aggregated by year.
    
    Args:
        spark: SparkSession object
        order_fact_df: DataFrame containing order fact data with order_date and profit columns
    
    Returns:
        DataFrame with columns [year, total_profit] or None if calculation fails
    """
    try:
        # Create temporary view for SQL querying
        order_fact_df.createOrReplaceTempView("temp_order_fact")
        # Execute SQL to sum profit by year, ordered from most recent to oldest
        result = spark.sql("""
            SELECT 
                YEAR(CAST(order_date AS DATE)) as year,  -- Extract year from order date
                ROUND(SUM(profit), 2) as total_profit  -- Sum profits and round to 2 decimals
            FROM temp_order_fact
            GROUP BY YEAR(CAST(order_date AS DATE))  -- Group by year
            ORDER BY year DESC  -- Sort by year descending (newest first)
        """)
        return result
    except Exception as e:
        print(f"Error in profit_by_year: {e}")
        return None  # Return None if calculation fails


def profit_by_year_and_category(spark, order_fact_df):
    """
    Calculate total profit aggregated by year and product category.
    
    Args:
        spark: SparkSession object
        order_fact_df: DataFrame containing order fact data with order_date, profit, and category columns
    
    Returns:
        DataFrame with columns [year, product_category, total_profit] or None if calculation fails
    """
    try:
        # Create temporary view for SQL querying
        order_fact_df.createOrReplaceTempView("temp_order_fact")
        # Execute SQL to sum profit by year and category, filtering out null categories
        result = spark.sql("""
            SELECT 
                YEAR(CAST(order_date AS DATE)) as year,  -- Extract year from order date
                category as product_category,  -- Product category
                ROUND(SUM(profit), 2) as total_profit  -- Sum profits and round to 2 decimals
            FROM temp_order_fact
            WHERE category IS NOT NULL  -- Filter out records with no category
            GROUP BY YEAR(CAST(order_date AS DATE)), category  -- Group by year and category
            ORDER BY year DESC, product_category  -- Sort by year desc, then category
        """)
        return result
    except Exception as e:
        print(f"Error in profit_by_year_and_category: {e}")
        return None  # Return None if calculation fails


def profit_by_customer(spark, order_fact_df):
    """
    Calculate total profit aggregated by customer.
    
    Args:
        spark: SparkSession object
        order_fact_df: DataFrame containing order fact data with customer_name and profit columns
    
    Returns:
        DataFrame with columns [customer, total_profit] or None if calculation fails
    """
    try:
        # Create temporary view for SQL querying
        order_fact_df.createOrReplaceTempView("temp_order_fact")
        # Execute SQL to sum profit by customer, sorted by highest profit first
        result = spark.sql("""
            SELECT 
                customer_name as customer,  -- Customer name
                ROUND(SUM(profit), 2) as total_profit  -- Sum profits and round to 2 decimals
            FROM temp_order_fact
            GROUP BY customer_name  -- Group by customer
            ORDER BY total_profit DESC  -- Sort by total profit descending (most profitable customers first)
        """)
        return result
    except Exception as e:
        print(f"Error in profit_by_customer: {e}")
        return None  # Return None if calculation fails


def profit_by_customer_and_year(spark, order_fact_df):
    """
    Calculate total profit aggregated by customer and year.
    
    Args:
        spark: SparkSession object
        order_fact_df: DataFrame containing order fact data with customer_name, order_date, and profit columns
    
    Returns:
        DataFrame with columns [year, customer, total_profit] or None if calculation fails
    """
    try:
        # Create temporary view for SQL querying
        order_fact_df.createOrReplaceTempView("temp_order_fact")
        # Execute SQL to sum profit by customer and year, showing profit trends over time
        result = spark.sql("""
            SELECT 
                YEAR(CAST(order_date AS DATE)) as year,  -- Extract year from order date
                customer_name as customer,  -- Customer name
                ROUND(SUM(profit), 2) as total_profit  -- Sum profits and round to 2 decimals
            FROM temp_order_fact
            GROUP BY YEAR(CAST(order_date AS DATE)), customer_name  -- Group by year and customer
            ORDER BY year DESC, total_profit DESC  -- Sort by year desc, then profit desc
        """)
        return result
    except Exception as e:
        print(f"Error in profit_by_customer_and_year: {e}")
        return None  # Return None if calculation fails