"""
Unit tests for the Customers module.

Tests cover:
- Initialization and configuration
- Reading customer data from Excel files
- Writing raw and enriched customer data
- Schema validation and data type conversions
- Error handling for missing/malformed data
"""
import pytest
import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from Customers import Customers


class TestCustomersInit:
    """Test Customers class initialization and setup."""
    
    def test_customers_init(self, spark_session, temp_data_dir):
        """Test that Customers class initializes correctly with required dependencies.
        
        Verifies that the Spark session and data directory path are properly stored.
        """
        customers = Customers(spark_session, temp_data_dir)
        assert customers.spark is not None  # Spark session should be initialized
        assert customers.data_dir == temp_data_dir  # Data directory should match input
    
    def test_customers_init_with_valid_spark(self, spark_session, temp_data_dir):
        """Test Customers initialization preserves the Spark session object.
        
        Verifies that the same Spark session instance is stored, not a copy.
        """
        customers = Customers(spark_session, temp_data_dir)
        assert customers.spark == spark_session  # Should be same object reference


class TestCustomersReadWriteRaw:
    """Test reading and writing raw customer data."""
    
    def test_write_customers_raw_success(self, spark_session, temp_data_dir, sample_customers_df):
        """Test successfully writing customers data to raw parquet format.
        
        Verifies:
        - Function returns True on success
        - Parquet file is created in raw/ directory
        """
        customers = Customers(spark_session, temp_data_dir)
        result = customers.write_customers_raw(sample_customers_df)
        
        assert result is True  # Should return True on success
        # Verify the parquet file was created
        parquet_path = os.path.join(temp_data_dir, "raw", "customers.parquet")
        assert os.path.exists(parquet_path)  # File should exist
    
    def test_write_customers_raw_creates_directory_structure(self, spark_session, temp_data_dir, sample_customers_df):
        """Test that write_customers_raw creates necessary directory structure.
        
        Verifies that the raw/ directory exists after writing data.
        """
        customers = Customers(spark_session, temp_data_dir)
        customers.write_customers_raw(sample_customers_df)
        
        raw_path = os.path.join(temp_data_dir, "raw")
        assert os.path.isdir(raw_path)  # raw/ directory should exist
    
    def test_write_customers_raw_with_empty_dataframe(self, spark_session, temp_data_dir):
        """Test writing empty customer DataFrame."""
        from pyspark.sql import types as T
        
        customers = Customers(spark_session, temp_data_dir)
        empty_schema = T.StructType([
            T.StructField("customer_id", T.StringType(), True),
            T.StructField("customer_name", T.StringType(), True),
        ])
        empty_df = spark_session.createDataFrame([], schema=empty_schema)
        
        result = customers.write_customers_raw(empty_df)
        assert result is True


class TestCustomersReadWriteEnriched:
    """Test reading and writing enriched customer data."""
    
    def test_write_customers_enriched_success(self, spark_session, temp_data_dir, sample_customers_df):
        """Test successfully writing enriched customer data."""
        customers = Customers(spark_session, temp_data_dir)

        sample_customers_df = (
            sample_customers_df
            .withColumnRenamed("customer_id", "Customer ID")
            .withColumnRenamed("customer_name", "Customer Name")
            .withColumnRenamed("email", "email")
            .withColumnRenamed("phone", "phone")
            .withColumnRenamed("address", "address")
            .withColumnRenamed("segment", "Segment")
            .withColumnRenamed("country", "Country")
            .withColumnRenamed("city", "City")
            .withColumnRenamed("state", "State")
            .withColumnRenamed("postal_code", "Postal Code")
            .withColumnRenamed("region", "Region")
        )

        result = customers.write_customers_enriched(sample_customers_df)
        
        assert result is True
        # Verify the parquet file was created
        parquet_path = os.path.join(temp_data_dir, "enriched", "customers.parquet")
        assert os.path.exists(parquet_path)
    
    def test_write_customers_enriched_transforms_columns(self, spark_session, temp_data_dir, sample_customers_df):
        """Test that enriched write properly transforms column names."""
        customers = Customers(spark_session, temp_data_dir)
        customers.write_customers_enriched(sample_customers_df)
        
        # Read back and verify column names were transformed
        enriched_df = customers.read_customers_enriched()
        assert "customer_id" in enriched_df.columns
        assert "customer_name" in enriched_df.columns
        assert "postal_code" in enriched_df.columns
    
    def test_write_customers_enriched_casts_postal_code(self, spark_session, temp_data_dir, sample_customers_df):
        """Test that postal_code is properly cast to IntegerType."""
        customers = Customers(spark_session, temp_data_dir)
        customers.write_customers_enriched(sample_customers_df)
        
        enriched_df = customers.read_customers_enriched()
        # Check the schema of postal_code column
        postal_code_field = next(f for f in enriched_df.schema if f.name == "postal_code")
        assert str(postal_code_field.dataType) == "IntegerType()"


class TestCustomersReadEnriched:
    """Test reading enriched customer data."""
    
    def test_read_customers_enriched_returns_dataframe(self, spark_session, temp_data_dir, sample_customers_df):
        """Test that read_customers_enriched returns a DataFrame."""
        customers = Customers(spark_session, temp_data_dir)
        customers.write_customers_enriched(sample_customers_df)
        
        result = customers.read_customers_enriched()
        assert result is not None
        assert hasattr(result, 'collect')  # It's a DataFrame
    
    def test_read_customers_enriched_preserves_data(self, spark_session, temp_data_dir, sample_customers_df):
        """Test that read_customers_enriched preserves data integrity."""
        customers = Customers(spark_session, temp_data_dir)

        sample_customers_df = (
            sample_customers_df
            .withColumnRenamed("customer_id", "Customer ID")
            .withColumnRenamed("customer_name", "Customer Name")
            .withColumnRenamed("email", "email")
            .withColumnRenamed("phone", "phone")
            .withColumnRenamed("address", "address")
            .withColumnRenamed("segment", "Segment")
            .withColumnRenamed("country", "Country")
            .withColumnRenamed("city", "City")
            .withColumnRenamed("state", "State")
            .withColumnRenamed("postal_code", "Postal Code")
            .withColumnRenamed("region", "Region")
        )

        customers.write_customers_enriched(sample_customers_df)
        
        enriched_df = customers.read_customers_enriched()
        
        # Check row count
        original_count = sample_customers_df.count()
        read_count = enriched_df.count()
        assert original_count == read_count
    
    def test_read_customers_enriched_correct_schema(self, spark_session, temp_data_dir, sample_customers_df):
        """Test that read_customers_enriched has correct schema."""
        customers = Customers(spark_session, temp_data_dir)
        customers.write_customers_enriched(sample_customers_df)
        
        enriched_df = customers.read_customers_enriched()
        column_names = enriched_df.columns
        
        expected_columns = [
            "customer_id", "customer_name", "email", "phone", "address",
            "segment", "country", "city", "state", "postal_code", "region"
        ]
        for col in expected_columns:
            assert col in column_names
    
    def test_read_customers_enriched_nonexistent_file(self, spark_session, temp_data_dir):
        """Test read_customers_enriched returns empty DataFrame if file doesn't exist."""
        customers = Customers(spark_session, temp_data_dir)
        
        # Try to read without writing first
        result = customers.read_customers_enriched()
        assert result.count() == 0
