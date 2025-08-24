import duckdb

# Path to your DuckDB database
db_path = '/Users/shubhammantri/Downloads/random/freelancing_work/sample_data.duckdb'

# Connect to the DuckDB database
conn = duckdb.connect(database=db_path, read_only=True)

# Run the query
query = """
WITH customer_2023_activity AS (
    -- Get last purchase date for each customer in 2023
    SELECT 
        customer_id,
        MAX(order_date) as last_2023_purchase
    FROM orders 
    WHERE EXTRACT(YEAR FROM order_date) = 2023
    GROUP BY customer_id
),
dormant_customers AS (
    -- Find customers who went dormant (3+ months gap after last 2023 purchase)
    SELECT 
        customer_id,
        last_2023_purchase
    FROM customer_2023_activity
    WHERE last_2023_purchase <= '2023-09-30'  -- Must have last purchase by Sept 30 to allow 3+ month gap
),
customer_2024_reactivation AS (
    -- Get first purchase date for dormant customers who reactivated in 2024
    SELECT 
        o.customer_id,
        MIN(o.order_date) as first_2024_purchase
    FROM orders o
    JOIN dormant_customers dc ON o.customer_id = dc.customer_id
    WHERE EXTRACT(YEAR FROM o.order_date) = 2024
    GROUP BY o.customer_id
),
reactivation_with_category AS (
    -- Get the category of the first 2024 purchase for each reactivated customer
    SELECT DISTINCT
        r24.customer_id,
        r24.first_2024_purchase,
        FIRST_VALUE(p.category) OVER (PARTITION BY r24.customer_id ORDER BY p.product_id) as reactivation_category
    FROM customer_2024_reactivation r24
    JOIN orders o ON r24.customer_id = o.customer_id AND r24.first_2024_purchase = o.order_date
    JOIN products p ON o.product_id = p.product_id
),
reactivated_customers AS (
    -- Calculate dormancy period and get reactivation details
    SELECT 
        dc.customer_id,
        dc.last_2023_purchase,
        rwc.first_2024_purchase,
        rwc.reactivation_category,
        (rwc.first_2024_purchase - dc.last_2023_purchase) as dormancy_days
    FROM dormant_customers dc
    JOIN reactivation_with_category rwc ON dc.customer_id = rwc.customer_id
    WHERE (rwc.first_2024_purchase - dc.last_2023_purchase) >= 90  -- 3+ months
)
SELECT 
    COUNT(DISTINCT customer_id) as reactivated_customers_count,
    ROUND(AVG(dormancy_days), 1) as avg_dormancy_days,
    ROUND(AVG(dormancy_days)/30.0, 1) as avg_dormancy_months,
    reactivation_category,
    COUNT(*) as customers_in_category
FROM reactivated_customers
GROUP BY reactivation_category
ORDER BY customers_in_category DESC;
"""
result = conn.execute(query).fetchdf()  # fetches result as a pandas DataFrame

# Display the result
print(result)

# Close the connection
conn.close()
