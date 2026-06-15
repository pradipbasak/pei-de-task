"""
Pytest configuration and shared fixtures for all tests.

This module provides:
- spark_session: PySpark session for distributed testing
- temp_data_dir: Temporary directory structure matching production layout
- sample_*_df: Pre-built test data DataFrames for consistent testing
"""
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql.functions import lit
import tempfile
import os
import sys
from datetime import date


@pytest.fixture(scope="session")
def spark_session():
    """Create a Spark session for testing.
    
    Session scope means this fixture is created once per test session and reused
    for all tests. This improves test performance by avoiding repeated Spark initialization.
    
    Returns:
        SparkSession configured for local mode with minimal shuffle partitions
    """
    # Initialize Spark with local master and minimal resource overhead
    spark = SparkSession.builder \
        .appName("pytest") \
        .master("local") \
        .config("spark.sql.shuffle.partitions", "1") \
        .getOrCreate()
    yield spark  # Provide spark to tests
    spark.stop()  # Clean up after all tests complete


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data with production-like structure.
    
    Auto-cleanup occurs when the fixture context exits. Each test gets its own
    isolated temporary directory to avoid test interference.
    
    Yields:
        str: Path to temporary directory with subdirectories: files/, raw/, enriched/, aggregated/
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create necessary subdirectories matching the production data structure
        # files/: Source data location (Excel, CSV, JSON)
        # raw/: Raw data after first read from source files
        # enriched/: Transformed data with standardized column names
        # aggregated/: Aggregated metrics and summary data
        os.makedirs(os.path.join(tmpdir, "files"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "enriched"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "aggregated"), exist_ok=True)
        yield tmpdir


@pytest.fixture
def sample_orders_df(spark_session):
    """Create a sample orders DataFrame for testing.
    
    Provides consistent test data with orders spanning multiple years (2015-2017)
    and various profit scenarios (positive, negative, zero).
    
    Args:
        spark_session: Spark session fixture for DataFrame creation
    
    Returns:
        DataFrame: Sample orders with 3 rows containing realistic test scenarios
    """
    schema = T.StructType([
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
    
    # Test data with varied scenarios:
    # Row 1: Positive profit with discount (2016, $15.5 profit)
    # Row 2: Good profit with no discount (2017, $50.0 profit)
    # Row 3: Negative profit/loss with high discount (2015, -$10.5 loss)
    data = [
        (1, "ORD-001", date(2016, 1, 15), date(2016, 1, 20), "Standard", "CUST-001", "PROD-001", 2, 100.0, 0.1, 15.5),
        (2, "ORD-002", date(2017, 3, 20), date(2017, 3, 25), "Express", "CUST-002", "PROD-002", 1, 250.0, 0.0, 50.0),
        (3, "ORD-003", date(2015, 6, 10), date(2015, 6, 15), "Standard", "CUST-001", "PROD-003", 3, 75.0, 0.2, -10.5),
    ]
    
    return spark_session.createDataFrame(data, schema=schema)


@pytest.fixture
def sample_customers_df(spark_session):
    """Create a sample customers DataFrame for testing.
    
    Provides test customer data with different segments and geographic locations
    to test joins and filtering logic.
    
    Args:
        spark_session: Spark session fixture for DataFrame creation
    
    Returns:
        DataFrame: Sample customers with 2 rows representing different segments
    """
    schema = T.StructType([
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
    
    # Test data with different customer segments:
    # Customer 1: Consumer segment, East region, NY
    # Customer 2: Corporate segment, West region, CA
    data = [
        ("CUST-001", "John Doe", "john@example.com", "555-0001", "123 Main St", "Consumer", "USA", "New York", "NY", 10001, "East"),
        ("CUST-002", "Jane Smith", "jane@example.com", "555-0002", "456 Oak Ave", "Corporate", "USA", "Los Angeles", "CA", 90001, "West"),
    ]
    
    return spark_session.createDataFrame(data, schema=schema)


@pytest.fixture
def sample_products_df(spark_session):
    """Create a sample products DataFrame for testing.
    
    Provides test product data with different categories and price points
    to test joins and aggregation logic.
    
    Args:
        spark_session: Spark session fixture for DataFrame creation
    
    Returns:
        DataFrame: Sample products with 3 rows from different categories
    """
    schema = T.StructType([
        T.StructField("product_id", T.StringType(), True),
        T.StructField("category", T.StringType(), True),
        T.StructField("sub_category", T.StringType(), True),
        T.StructField("product_name", T.StringType(), True),
        T.StructField("state", T.StringType(), True),
        T.StructField("price_per_product", T.DoubleType(), True),
    ])
    
    # Test data from different product categories:
    # Product 1: Furniture category, Chairs subcategory, higher price point ($150)
    # Product 2: Technology category, Accessories, lower price point ($15)
    # Product 3: Office Supplies category, Paper, lowest price point ($5)
    data = [
        ("PROD-001", "Furniture", "Chairs", "Office Chair", "NY", 150.0),
        ("PROD-002", "Technology", "Accessories", "USB Cable", "CA", 15.0),
        ("PROD-003", "Office Supplies", "Paper", "Copy Paper", "TX", 5.0),
    ]
    
    return spark_session.createDataFrame(data, schema=schema)


@pytest.fixture
def sample_order_fact_df(spark_session, sample_orders_df, sample_customers_df, sample_products_df):
    """Create a sample order fact DataFrame for testing.
    
    Simulates the production order_fact table by joining orders with customers
    and products. This is the main table used by aggregation tests.
    
    Args:
        spark_session: Spark session fixture
        sample_orders_df: Sample orders DataFrame
        sample_customers_df: Sample customers DataFrame
        sample_products_df: Sample products DataFrame
    
    Returns:
        DataFrame: Combined fact table with order, customer, and product dimensions
    """
    # Create fact table by left-joining dimension tables
    # Left join preserves all orders even if customer or product data is missing
    order_fact = (
        sample_orders_df
        .join(sample_customers_df, "customer_id", "left")  # Join with customers
        .join(sample_products_df, "product_id", "left")    # Join with products
        .select(
            # Order dimensions
            sample_orders_df.order_id,
            sample_orders_df.order_date,
            sample_orders_df.ship_date,
            sample_orders_df.ship_mode,
            sample_orders_df.quantity,
            sample_orders_df.price,
            sample_orders_df.discount,
            sample_orders_df.profit,
            # Customer dimensions
            sample_customers_df.customer_id,
            sample_customers_df.customer_name,
            sample_customers_df.country,
            # Product dimensions
            sample_products_df.product_id,
            sample_products_df.category,
            sample_products_df.sub_category
        )
    )
    return order_fact
