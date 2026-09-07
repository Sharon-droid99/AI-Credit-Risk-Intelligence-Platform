-- ============================================================
-- CREDIT RISK PLATFORM - DUCKDB SCHEMA
-- ============================================================

-- Main applicant-level analytical table
CREATE OR REPLACE TABLE applicants AS
SELECT *
FROM read_csv_auto(
    'models/application_features.csv',
    HEADER = TRUE
);