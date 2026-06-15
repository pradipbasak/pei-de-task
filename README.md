## Tree View

```
pei-de-task/
├── assets/
│   └── style.css
├── data/
│   ├── aggregated/ # contains agregated profit table
│   │   └── profit_info.parquet/
│   ├── enriched/ # contains all the enriched tables
│   │   ├── customers.parquet/
│   │   ├── order_fact.parquet/
│   │   ├── orders.parquet/
│   │   └── products.parquet/
│   ├── files/ # these are the source files
│   │   ├── Customer.xlsx
│   │   ├── Orders.json
│   │   └── Products.csv
│   └── raw/ # contains tabular version of the source files
│       ├── customers.parquet/
│       ├── orders.parquet/
│       └── products.parquet/
├── pytest.ini
├── report.html
├── requirements.txt
├── src/
│   ├── Customers.py
│   ├── Orders.py
│   ├── Products.py
│   ├── __init__.py
│   ├── etl.ipynb # main orchestration file
│   └── utils.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_customers.py
    ├── test_orders.py
    ├── test_products.py
    └── test_utils.py

```