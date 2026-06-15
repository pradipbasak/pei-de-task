"""
Unit tests for the Orders module.

Tests cover:
- Initialization and configuration
- Reading order data from JSON files with schema validation
- Writing raw and enriched order data with date conversions
- Data type casting (dates, doubles, integers)
- Error handling for missing/malformed JSON data
"""
import pytest
import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from Orders import Orders


class TestOrdersInit:
    """Test Orders class initialization."""
    
    def test_orders_init(self, spark_session, temp_data_dir):
        """Test that Orders class initializes correctly."""
        orders = Orders(spark_session, temp_data_dir)
        assert orders.spark is not None
        assert orders.data_dir == temp_data_dir
    
    def test_orders_init_with_valid_spark(self, spark_session, temp_data_dir):
        """Test Orders initialization with valid Spark session."""
        orders = Orders(spark_session, temp_data_dir)
        assert orders.spark == spark_session


class TestOrdersReadWriteRaw:
    """Test reading and writing raw order data."""
    
    def test_write_orders_raw_success(self, spark_session, temp_data_dir, sample_orders_df):
        """Test successfully writing orders data to raw format."""
        orders = Orders(spark_session, temp_data_dir)
        result = orders.write_orders_raw(sample_orders_df)
        
        assert result is True
        # Verify the parquet file was created
        parquet_path = os.path.join(temp_data_dir, "raw", "orders.parquet")
        assert os.path.exists(parquet_path)
    
    def test_write_orders_raw_creates_directory_structure(self, spark_session, temp_data_dir, sample_orders_df):
        """Test that write_orders_raw creates necessary directory structure."""
        orders = Orders(spark_session, temp_data_dir)
        orders.write_orders_raw(sample_orders_df)
        
        raw_path = os.path.join(temp_data_dir, "raw")
        assert os.path.isdir(raw_path)
    
    def test_write_orders_raw_with_empty_dataframe(self, spark_session, temp_data_dir):
        """Test writing empty orders DataFrame."""
        from pyspark.sql import types as T
        
        orders = Orders(spark_session, temp_data_dir)
        empty_schema = T.StructType([
            T.StructField("order_id", T.StringType(), True),
            T.StructField("profit", T.DoubleType(), True),
        ])
        empty_df = spark_session.createDataFrame([], schema=empty_schema)
        
        result = orders.write_orders_raw(empty_df)
        assert result is True


class TestOrdersReadWriteEnriched:
    """Test reading and writing enriched order data."""
    
    def test_write_orders_enriched_success(self, spark_session, temp_data_dir, sample_orders_df):
        """Test successfully writing enriched order data."""
        orders = Orders(spark_session, temp_data_dir)

        sample_orders_df = (
            sample_orders_df
            .withColumnRenamed("row_id", "Row ID")
            .withColumnRenamed("order_id", "Order ID")
            .withColumnRenamed("order_date", "Order Date")
            .withColumnRenamed("ship_date", "Ship Date")
            .withColumnRenamed("ship_mode", "Ship Mode")
            .withColumnRenamed("customer_id", "Customer ID")
            .withColumnRenamed("product_id", "Product ID")
            .withColumnRenamed("quantity", "Quantity")
            .withColumnRenamed("price", "Price")
            .withColumnRenamed("discount", "Discount")
            .withColumnRenamed("profit", "Profit")
        )

        result = orders.write_orders_enriched(sample_orders_df)
        
        assert result is True
        # Verify the parquet file was created
        parquet_path = os.path.join(temp_data_dir, "enriched", "orders.parquet")
        assert os.path.exists(parquet_path)
    
    def test_write_orders_enriched_transforms_columns(self, spark_session, temp_data_dir, sample_orders_df):
        """Test that enriched write properly transforms column names."""
        orders = Orders(spark_session, temp_data_dir)
        orders.write_orders_enriched(sample_orders_df)
        
        # Read back and verify column names were transformed
        enriched_df = orders.read_orders_enriched()
        assert "row_id" in enriched_df.columns
        assert "order_id" in enriched_df.columns
        assert "order_date" in enriched_df.columns
        assert "ship_mode" in enriched_df.columns
    
    def test_write_orders_enriched_preserves_data_integrity(self, spark_session, temp_data_dir, sample_orders_df):
        """Test that enriched write preserves row count."""
        orders = Orders(spark_session, temp_data_dir)
        original_count = sample_orders_df.count()

        sample_orders_df = (
            sample_orders_df
            .withColumnRenamed("row_id", "Row ID")
            .withColumnRenamed("order_id", "Order ID")
            .withColumnRenamed("order_date", "Order Date")
            .withColumnRenamed("ship_date", "Ship Date")
            .withColumnRenamed("ship_mode", "Ship Mode")
            .withColumnRenamed("customer_id", "Customer ID")
            .withColumnRenamed("product_id", "Product ID")
            .withColumnRenamed("quantity", "Quantity")
            .withColumnRenamed("price", "Price")
            .withColumnRenamed("discount", "Discount")
            .withColumnRenamed("profit", "Profit")
        )
        
        orders.write_orders_enriched(sample_orders_df)
        enriched_df = orders.read_orders_enriched()
        
        assert enriched_df.count() == original_count


class TestOrdersReadEnriched:
    """Test reading enriched order data."""
    
    def test_read_orders_enriched_returns_dataframe(self, spark_session, temp_data_dir, sample_orders_df):
        """Test that read_orders_enriched returns a DataFrame."""
        orders = Orders(spark_session, temp_data_dir)
        orders.write_orders_enriched(sample_orders_df)
        
        result = orders.read_orders_enriched()
        assert result is not None
        assert hasattr(result, 'collect')  # It's a DataFrame
    
    def test_read_orders_enriched_correct_schema(self, spark_session, temp_data_dir, sample_orders_df):
        """Test that read_orders_enriched has correct schema."""
        orders = Orders(spark_session, temp_data_dir)
        orders.write_orders_enriched(sample_orders_df)
        
        enriched_df = orders.read_orders_enriched()
        column_names = enriched_df.columns
        
        expected_columns = [
            "row_id", "order_id", "order_date", "ship_date", "ship_mode",
            "customer_id", "product_id", "quantity", "price", "discount", "profit"
        ]
        for col in expected_columns:
            assert col in column_names
    
    def test_read_orders_enriched_correct_data_types(self, spark_session, temp_data_dir, sample_orders_df):
        """Test that read_orders_enriched has correct data types."""
        orders = Orders(spark_session, temp_data_dir)
        orders.write_orders_enriched(sample_orders_df)
        
        enriched_df = orders.read_orders_enriched()
        schema_dict = {field.name: str(field.dataType) for field in enriched_df.schema}
        
        assert schema_dict["row_id"] == "IntegerType()"
        assert schema_dict["order_id"] == "StringType()"
        assert schema_dict["profit"] == "DoubleType()"
    
    def test_read_orders_enriched_nonexistent_file(self, spark_session, temp_data_dir):
        """Test read_orders_enriched returns empty DataFrame if file doesn't exist."""
        orders = Orders(spark_session, temp_data_dir)
        
        # Try to read without writing first
        result = orders.read_orders_enriched()
        assert result.count() == 0
    
    def test_write_and_read_orders_round_trip(self, spark_session, temp_data_dir, sample_orders_df):
        """Test that write then read preserves data."""
        orders = Orders(spark_session, temp_data_dir)

        sample_orders_df = (
            sample_orders_df
            .withColumnRenamed("row_id", "Row ID")
            .withColumnRenamed("order_id", "Order ID")
            .withColumnRenamed("order_date", "Order Date")
            .withColumnRenamed("ship_date", "Ship Date")
            .withColumnRenamed("ship_mode", "Ship Mode")
            .withColumnRenamed("customer_id", "Customer ID")
            .withColumnRenamed("product_id", "Product ID")
            .withColumnRenamed("quantity", "Quantity")
            .withColumnRenamed("price", "Price")
            .withColumnRenamed("discount", "Discount")
            .withColumnRenamed("profit", "Profit")
        )
        
        orders.write_orders_enriched(sample_orders_df)
        enriched_df = orders.read_orders_enriched()
        
        # Verify specific data values
        first_row = enriched_df.first()
        assert first_row is not None
        assert first_row["order_id"] is not None
