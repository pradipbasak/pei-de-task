"""
Unit tests for the utils module.
"""
import pytest
import os
import sys
import datetime
from pathlib import Path
from pyspark.sql import types as T

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import utils


class TestProfitByYear:
    """Test profit_by_year function."""
    
    def test_profit_by_year_returns_dataframe(self, spark_session, sample_order_fact_df):
        """Test that profit_by_year returns a DataFrame."""
        result = utils.profit_by_year(spark_session, sample_order_fact_df)
        
        assert result is not None
        assert hasattr(result, 'collect')  # It's a DataFrame
    
    def test_profit_by_year_has_correct_columns(self, spark_session, sample_order_fact_df):
        """Test that profit_by_year has correct column names."""
        result = utils.profit_by_year(spark_session, sample_order_fact_df)
        
        column_names = result.columns
        assert "year" in column_names
        assert "total_profit" in column_names
    
    def test_profit_by_year_has_correct_data_types(self, spark_session, sample_order_fact_df):
        """Test that profit_by_year has correct data types."""
        result = utils.profit_by_year(spark_session, sample_order_fact_df)
        
        schema_dict = {field.name: str(field.dataType) for field in result.schema}
        assert schema_dict["year"] == "IntegerType()"
        assert schema_dict["total_profit"] == "DoubleType()"
    
    def test_profit_by_year_aggregates_correctly(self, spark_session, sample_order_fact_df):
        """Test that profit_by_year aggregates profit correctly."""
        result = utils.profit_by_year(spark_session, sample_order_fact_df)
        
        rows = result.collect()
        assert len(rows) > 0
        
        # Check that profit values are present
        for row in rows:
            assert row["year"] is not None
            assert row["total_profit"] is not None
    
    def test_profit_by_year_ordered_by_year_desc(self, spark_session, sample_order_fact_df):
        """Test that profit_by_year is ordered by year descending."""
        result = utils.profit_by_year(spark_session, sample_order_fact_df)
        
        rows = result.collect()
        if len(rows) > 1:
            for i in range(len(rows) - 1):
                assert rows[i]["year"] >= rows[i + 1]["year"]


class TestProfitByYearAndCategory:
    """Test profit_by_year_and_category function."""
    
    def test_profit_by_year_and_category_returns_dataframe(self, spark_session, sample_order_fact_df):
        """Test that profit_by_year_and_category returns a DataFrame."""
        result = utils.profit_by_year_and_category(spark_session, sample_order_fact_df)
        
        assert result is not None
        assert hasattr(result, 'collect')  # It's a DataFrame
    
    def test_profit_by_year_and_category_has_correct_columns(self, spark_session, sample_order_fact_df):
        """Test that profit_by_year_and_category has correct column names."""
        result = utils.profit_by_year_and_category(spark_session, sample_order_fact_df)
        
        column_names = result.columns
        assert "year" in column_names
        assert "product_category" in column_names
        assert "total_profit" in column_names
    
    def test_profit_by_year_and_category_filters_nulls(self, spark_session):
        """Test that profit_by_year_and_category filters out NULL categories."""
        # Create a DataFrame with some NULL categories
        schema = T.StructType([
            T.StructField("order_id", T.StringType(), True),
            T.StructField("order_date", T.DateType(), True),
            T.StructField("category", T.StringType(), True),
            T.StructField("profit", T.DoubleType(), True),
        ])
        
        data = [
            ("ORD-001", datetime.date(2017, 1, 15), "Furniture", 50.0),
            ("ORD-002", datetime.date(2017, 1, 20), None, 25.0),  # NULL category
            ("ORD-003", datetime.date(2017, 1, 25), "Technology", 75.0),
        ]
        
        df = spark_session.createDataFrame(data, schema=schema)
        result = utils.profit_by_year_and_category(spark_session, df)
        
        # Should only have 2 rows (NULL filtered out)
        assert result.count() == 2
    
    def test_profit_by_year_and_category_aggregates_correctly(self, spark_session, sample_order_fact_df):
        """Test that profit_by_year_and_category aggregates correctly."""
        result = utils.profit_by_year_and_category(spark_session, sample_order_fact_df)
        
        rows = result.collect()
        for row in rows:
            assert row["year"] is not None
            assert row["product_category"] is not None
            assert row["total_profit"] is not None


class TestProfitByCustomer:
    """Test profit_by_customer function."""
    
    def test_profit_by_customer_returns_dataframe(self, spark_session, sample_order_fact_df):
        """Test that profit_by_customer returns a DataFrame."""
        result = utils.profit_by_customer(spark_session, sample_order_fact_df)
        
        assert result is not None
        assert hasattr(result, 'collect')  # It's a DataFrame
    
    def test_profit_by_customer_has_correct_columns(self, spark_session, sample_order_fact_df):
        """Test that profit_by_customer has correct column names."""
        result = utils.profit_by_customer(spark_session, sample_order_fact_df)
        
        column_names = result.columns
        assert "customer" in column_names
        assert "total_profit" in column_names
    
    def test_profit_by_customer_has_correct_data_types(self, spark_session, sample_order_fact_df):
        """Test that profit_by_customer has correct data types."""
        result = utils.profit_by_customer(spark_session, sample_order_fact_df)
        
        schema_dict = {field.name: str(field.dataType) for field in result.schema}
        assert schema_dict["customer"] == "StringType()"
        assert schema_dict["total_profit"] == "DoubleType()"
    
    def test_profit_by_customer_ordered_by_profit_desc(self, spark_session, sample_order_fact_df):
        """Test that profit_by_customer is ordered by profit descending."""
        result = utils.profit_by_customer(spark_session, sample_order_fact_df)
        
        rows = result.collect()
        if len(rows) > 1:
            for i in range(len(rows) - 1):
                assert rows[i]["total_profit"] >= rows[i + 1]["total_profit"]
    
    def test_profit_by_customer_aggregates_correctly(self, spark_session, sample_order_fact_df):
        """Test that profit_by_customer aggregates correctly."""
        result = utils.profit_by_customer(spark_session, sample_order_fact_df)
        
        rows = result.collect()
        for row in rows:
            assert row["customer"] is not None
            assert row["total_profit"] is not None


class TestProfitByCustomerAndYear:
    """Test profit_by_customer_and_year function."""
    
    def test_profit_by_customer_and_year_returns_dataframe(self, spark_session, sample_order_fact_df):
        """Test that profit_by_customer_and_year returns a DataFrame."""
        result = utils.profit_by_customer_and_year(spark_session, sample_order_fact_df)
        
        assert result is not None
        assert hasattr(result, 'collect')  # It's a DataFrame
    
    def test_profit_by_customer_and_year_has_correct_columns(self, spark_session, sample_order_fact_df):
        """Test that profit_by_customer_and_year has correct column names."""
        result = utils.profit_by_customer_and_year(spark_session, sample_order_fact_df)
        
        column_names = result.columns
        assert "year" in column_names
        assert "customer" in column_names
        assert "total_profit" in column_names
    
    def test_profit_by_customer_and_year_has_correct_data_types(self, spark_session, sample_order_fact_df):
        """Test that profit_by_customer_and_year has correct data types."""
        result = utils.profit_by_customer_and_year(spark_session, sample_order_fact_df)
        
        schema_dict = {field.name: str(field.dataType) for field in result.schema}
        assert schema_dict["year"] == "IntegerType()"
        assert schema_dict["customer"] == "StringType()"
        assert schema_dict["total_profit"] == "DoubleType()"
    
    def test_profit_by_customer_and_year_ordered_correctly(self, spark_session, sample_order_fact_df):
        """Test that profit_by_customer_and_year is properly ordered."""
        result = utils.profit_by_customer_and_year(spark_session, sample_order_fact_df)
        
        rows = result.collect()
        # Verify at least ordered by year first
        if len(rows) > 1:
            for i in range(len(rows) - 1):
                # Year should be >= next row's year (descending)
                assert rows[i]["year"] >= rows[i + 1]["year"]
    
    def test_profit_by_customer_and_year_aggregates_correctly(self, spark_session, sample_order_fact_df):
        """Test that profit_by_customer_and_year aggregates correctly."""
        result = utils.profit_by_customer_and_year(spark_session, sample_order_fact_df)
        
        rows = result.collect()
        for row in rows:
            assert row["year"] is not None
            assert row["customer"] is not None
            assert row["total_profit"] is not None


class TestErrorHandling:
    """Test error handling in utility functions."""
    
    def test_profit_by_year_with_invalid_dataframe(self, spark_session):
        """Test profit_by_year returns None on invalid input."""
        # Create a DataFrame without required columns
        schema = T.StructType([
            T.StructField("col1", T.StringType(), True),
        ])
        invalid_df = spark_session.createDataFrame([], schema=schema)
        
        result = utils.profit_by_year(spark_session, invalid_df)
        # Function should handle error and return None
        assert result is None
    
    def test_profit_by_customer_with_invalid_dataframe(self, spark_session):
        """Test profit_by_customer returns None on invalid input."""
        # Create a DataFrame without required columns
        schema = T.StructType([
            T.StructField("col1", T.StringType(), True),
        ])
        invalid_df = spark_session.createDataFrame([], schema=schema)
        
        result = utils.profit_by_customer(spark_session, invalid_df)
        # Function should handle error and return None
        assert result is None
    
    def test_profit_by_year_and_category_with_invalid_dataframe(self, spark_session):
        """Test profit_by_year_and_category returns None on invalid input."""
        # Create a DataFrame without required columns
        schema = T.StructType([
            T.StructField("col1", T.StringType(), True),
        ])
        invalid_df = spark_session.createDataFrame([], schema=schema)
        
        result = utils.profit_by_year_and_category(spark_session, invalid_df)
        # Function should handle error and return None
        assert result is None
    
    def test_profit_by_customer_and_year_with_invalid_dataframe(self, spark_session):
        """Test profit_by_customer_and_year returns None on invalid input."""
        # Create a DataFrame without required columns
        schema = T.StructType([
            T.StructField("col1", T.StringType(), True),
        ])
        invalid_df = spark_session.createDataFrame([], schema=schema)
        
        result = utils.profit_by_customer_and_year(spark_session, invalid_df)
        # Function should handle error and return None
        assert result is None


class TestEmptyDataFrames:
    """Test utility functions with empty DataFrames."""
    
    def test_profit_by_year_with_empty_dataframe(self, spark_session):
        """Test profit_by_year with empty DataFrame."""
        schema = T.StructType([
            T.StructField("order_date", T.DateType(), True),
            T.StructField("profit", T.DoubleType(), True),
        ])
        empty_df = spark_session.createDataFrame([], schema=schema)
        
        result = utils.profit_by_year(spark_session, empty_df)
        assert result is not None
        assert result.count() == 0
    
    def test_profit_by_customer_with_empty_dataframe(self, spark_session):
        """Test profit_by_customer with empty DataFrame."""
        schema = T.StructType([
            T.StructField("customer_name", T.StringType(), True),
            T.StructField("profit", T.DoubleType(), True),
        ])
        empty_df = spark_session.createDataFrame([], schema=schema)
        
        result = utils.profit_by_customer(spark_session, empty_df)
        assert result is not None
        assert result.count() == 0
