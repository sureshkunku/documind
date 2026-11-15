# SQL Basics

## SELECT and WHERE
SELECT chooses which columns to return. WHERE filters rows before any grouping happens.

## Joins
An INNER JOIN returns only rows that have a match in both tables. A LEFT JOIN returns every row from the left table, with NULL for columns from the right table when there is no match.

## GROUP BY and HAVING
GROUP BY collects rows with the same value into groups so aggregate functions like COUNT, SUM and AVG can run per group. HAVING filters groups after aggregation, while WHERE filters rows before it.

## Indexes
An index speeds up reads on a column, like the index of a book. Indexes slow down inserts and updates slightly, because the index must be updated too.

## Window functions
A window function computes a value across related rows without collapsing them into one row. RANK() OVER (PARTITION BY region ORDER BY sales DESC) ranks sales within each region.