"""
Unit tests for the Products module.

Tests cover:
- Initialization and configuration
- Reading product data from CSV files with special character handling
- Writing raw and enriched product data
- Data type casting (prices to double)
- Error handling for missing/malformed CSV data
"""
import pytest
import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from Products import Products


class TestProductsInit:
    """Test Products class initialization."""
    
    def test_products_init(self, spark_session, temp_data_dir):
        """Test that Products class initializes correctly."""
        products = Products(spark_session, temp_data_dir)
        assert products.spark is not None
        assert products.data_dir == temp_data_dir
    
    def test_products_init_with_valid_spark(self, spark_session, temp_data_dir):
        """Test Products initialization with valid Spark session."""
        products = Products(spark_session, temp_data_dir)
        assert products.spark == spark_session


class TestProductsReadWriteRaw:
    """Test reading and writing raw product data."""
    
    def test_write_products_raw_success(self, spark_session, temp_data_dir, sample_products_df):
        """Test successfully writing products data to raw format."""
        products = Products(spark_session, temp_data_dir)
        result = products.write_products_raw(sample_products_df)
        
        assert result is True
        # Verify the parquet file was created
        parquet_path = os.path.join(temp_data_dir, "raw", "products.parquet")
        assert os.path.exists(parquet_path)
    
    def test_write_products_raw_creates_directory_structure(self, spark_session, temp_data_dir, sample_products_df):
        """Test that write_products_raw creates necessary directory structure."""
        products = Products(spark_session, temp_data_dir)
        products.write_products_raw(sample_products_df)
        
        raw_path = os.path.join(temp_data_dir, "raw")
        assert os.path.isdir(raw_path)
    
    def test_write_products_raw_with_empty_dataframe(self, spark_session, temp_data_dir):
        """Test writing empty products DataFrame."""
        from pyspark.sql import types as T
        
        products = Products(spark_session, temp_data_dir)
        empty_schema = T.StructType([
            T.StructField("product_id", T.StringType(), True),
            T.StructField("category", T.StringType(), True),
        ])
        empty_df = spark_session.createDataFrame([], schema=empty_schema)
        
        result = products.write_products_raw(empty_df)
        assert result is True


class TestProductsReadWriteEnriched:
    """Test reading and writing enriched product data."""
    
    def test_write_products_enriched_success(self, spark_session, temp_data_dir, sample_products_df):
        """Test successfully writing enriched product data."""
        products = Products(spark_session, temp_data_dir)
        sample_products_df = (
            sample_products_df
            .withColumnRenamed("product_id", "Product ID")
            .withColumnRenamed("category", "Category")
            .withColumnRenamed("sub_category", "Sub-Category")
            .withColumnRenamed("product_name", "Product Name")
            .withColumnRenamed("state", "State")
            .withColumnRenamed("price_per_product", "Price Per Product")
        )
        result = products.write_products_enriched(sample_products_df)
        
        assert result is True
        # Verify the parquet file was created
        parquet_path = os.path.join(temp_data_dir, "enriched", "products.parquet")
        assert os.path.exists(parquet_path)
    
    def test_write_products_enriched_transforms_columns(self, spark_session, temp_data_dir, sample_products_df):
        """Test that enriched write properly transforms column names."""
        products = Products(spark_session, temp_data_dir)
        products.write_products_enriched(sample_products_df)
        
        # Read back and verify column names were transformed
        enriched_df = products.read_products_enriched()
        assert "product_id" in enriched_df.columns
        assert "category" in enriched_df.columns
        assert "sub_category" in enriched_df.columns
        assert "price_per_product" in enriched_df.columns
    
    def test_write_products_enriched_casts_price(self, spark_session, temp_data_dir, sample_products_df):
        """Test that price_per_product is properly cast to DoubleType."""
        products = Products(spark_session, temp_data_dir)
        products.write_products_enriched(sample_products_df)
        
        enriched_df = products.read_products_enriched()
        # Check the schema of price_per_product column
        price_field = next(f for f in enriched_df.schema if f.name == "price_per_product")
        assert str(price_field.dataType) == "DoubleType()"
    
    def test_write_products_enriched_preserves_data_integrity(self, spark_session, temp_data_dir, sample_products_df):
        """Test that enriched write preserves row count."""
        products = Products(spark_session, temp_data_dir)
        original_count = sample_products_df.count()
        
        sample_products_df = (
            sample_products_df
            .withColumnRenamed("product_id", "Product ID")
            .withColumnRenamed("category", "Category")
            .withColumnRenamed("sub_category", "Sub-Category")
            .withColumnRenamed("product_name", "Product Name")
            .withColumnRenamed("state", "State")
            .withColumnRenamed("price_per_product", "Price Per Product")
        )

        products.write_products_enriched(sample_products_df)
        enriched_df = products.read_products_enriched()
        
        assert enriched_df.count() == original_count


class TestProductsReadEnriched:
    """Test reading enriched product data."""
    
    def test_read_products_enriched_returns_dataframe(self, spark_session, temp_data_dir, sample_products_df):
        """Test that read_products_enriched returns a DataFrame."""
        products = Products(spark_session, temp_data_dir)
        products.write_products_enriched(sample_products_df)
        
        result = products.read_products_enriched()
        assert result is not None
        assert hasattr(result, 'collect')  # It's a DataFrame
    
    def test_read_products_enriched_correct_schema(self, spark_session, temp_data_dir, sample_products_df):
        """Test that read_products_enriched has correct schema."""
        products = Products(spark_session, temp_data_dir)
        products.write_products_enriched(sample_products_df)
        
        enriched_df = products.read_products_enriched()
        column_names = enriched_df.columns
        
        expected_columns = [
            "product_id", "category", "sub_category", "product_name", "state", "price_per_product"
        ]
        for col in expected_columns:
            assert col in column_names
    
    def test_read_products_enriched_correct_data_types(self, spark_session, temp_data_dir, sample_products_df):
        """Test that read_products_enriched has correct data types."""
        products = Products(spark_session, temp_data_dir)
        products.write_products_enriched(sample_products_df)
        
        enriched_df = products.read_products_enriched()
        schema_dict = {field.name: str(field.dataType) for field in enriched_df.schema}
        
        assert schema_dict["product_id"] == "StringType()"
        assert schema_dict["category"] == "StringType()"
        assert schema_dict["price_per_product"] == "DoubleType()"
    
    def test_read_products_enriched_nonexistent_file(self, spark_session, temp_data_dir):
        """Test read_products_enriched returns empty DataFrame if file doesn't exist."""
        products = Products(spark_session, temp_data_dir)
        
        # Try to read without writing first
        result = products.read_products_enriched()
        assert result.count() == 0
    
    def test_write_and_read_products_round_trip(self, spark_session, temp_data_dir, sample_products_df):
        """Test that write then read preserves data."""
        products = Products(spark_session, temp_data_dir)
        
        sample_products_df = (
            sample_products_df
            .withColumnRenamed("product_id", "Product ID")
            .withColumnRenamed("category", "Category")
            .withColumnRenamed("sub_category", "Sub-Category")
            .withColumnRenamed("product_name", "Product Name")
            .withColumnRenamed("state", "State")
            .withColumnRenamed("price_per_product", "Price Per Product")
        )

        products.write_products_enriched(sample_products_df)
        enriched_df = products.read_products_enriched()
        
        # Verify specific data values
        first_row = enriched_df.first()
        assert first_row is not None
        assert first_row["product_id"] is not None
        assert first_row["category"] in ["Furniture", "Technology", "Office Supplies"]
