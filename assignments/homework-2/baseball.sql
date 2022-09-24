# BDA602 : Assignment 2
# Problem Statement: To calculate Historic, Annual & Rolling - Batting Averages for all players.

# Extras:
# (1) Variable Datatype: I've defined variable names for each CREATE statement of temporary/non-temporary table.
#   - Reason being, for example, 'int' takes more space than 'tinyint'. Hence, defined data-types as per requirement.
# (2) Temporary table: Used for calling and referencing at multiple places to optimise query.
# (3) Indexes: I've used indexing on temporary/non-temporary tables for optimizing query.
# (4) Linters: I've used 'sql fluff' as my linter for the assignment.
# (5) Engine: I've used MyISAM Engine for my code.


# References:
# (a) Batting Average: https://en.wikipedia.org/wiki/Batting_average_(baseball)
# (b) MariaDB CREATE TABLE: https://mariadb.com/kb/en/create-table/#create-table-select
# (c) MyISAM Engine: https://mariadb.com/kb/en/show-engines/
# (d) SQL Fluff Linter: https://www.sqlfluff.com/


# (1) Historic Batting Average for every player:
# I observed, few rows in table batter_counts had few 'atBat' values as '0'.
# So to eliminate the Division by '0' error (infinity error), I added the condition of 'atBat' > 0.
DROP TABLE IF EXISTS player_historic_average;

CREATE TABLE player_historic_average
(batter mediumint, atBat smallint, Hit smallint, historic_batting_average float,
INDEX batter_idx (batter)) ENGINE=MyISAM

        SELECT
                batter,
                SUM(Hit) AS Hit,
                SUM( IFNULL(atBat,0) ) AS atBat,
                ROUND( SUM(Hit)/SUM(atBat) , 3 ) AS historic_batting_average
        FROM batter_counts
        WHERE atBat > 0
        GROUP BY batter
;

# Results: Historic Batting Average:
SELECT * FROM player_historic_average
LIMIT 0, 20;


# Temporary Table: for showing total hits and at bats for each player per game on every single date.
DROP TEMPORARY TABLE IF EXISTS temp_player_game_details;

CREATE TEMPORARY TABLE temp_player_game_details
(batter mediumint, game_id smallint, local_date date, atBat tinyint, Hit tinyint,
INDEX batter_idx (batter), INDEX game_idx (game_id), INDEX date_idx (local_date)) ENGINE=MyISAM

	    SELECT
                bc.batter,
                bc.game_id,
                DATE(g.local_date) AS local_date,
                bc.atBat,
                bc.Hit
        FROM batter_counts bc
        JOIN game g ON bc.game_id = g.game_id
        WHERE bc.atBat > 0
;


# (2) Annual Batting Average for every player:
# Referring to Temporary table to create a resultant table from its results.
DROP TABLE IF EXISTS player_annual_average;

CREATE TABLE player_annual_average
(batter mediumint, local_date date, annual_batting_average float,
INDEX batter_idx (batter), INDEX date_idx (local_date)) ENGINE=MyISAM

        SELECT
                batter,
                YEAR(local_date) AS year,
                ROUND( SUM(Hit)/SUM(atBat) , 3 ) AS annual_batting_average
        FROM temp_player_game_details
        GROUP BY batter, YEAR(local_date)
;

# Results: Annual Batting Average
SELECT batter, year, annual_batting_average FROM player_annual_average
LIMIT 0, 20;


# (3) Rolling (Last 100 days) Batting Average for every player:
# I'm dividing this calculation into two parts:
# (a) Find Rolling 100 days date fields and its Batting Average, using temporary tables.
# (b) Joining back the date to our batter-game table, to get batter & game wise results. (using self-join hint here)
DROP TEMPORARY TABLE IF EXISTS temp_rolling_dates;

CREATE TEMPORARY TABLE temp_rolling_dates
(batter mediumint, local_date date, atBat smallint, Hit smallint,
INDEX batter_idx (batter), INDEX date_idx (local_date)) ENGINE=MyISAM

        SELECT
                temp_1.batter,
                temp_1.local_date,
                SUM(temp_2.Hit) AS Hit,
                SUM(temp_2.atBat) AS atBat
                # COUNT(*) as count_rows
        FROM temp_player_game_details temp_1
        JOIN temp_player_game_details temp_2
                ON temp_2.local_date
                        BETWEEN DATE_SUB(temp_1.local_date, INTERVAL 100 DAY)
                            # Hint used below: Exclude Today's date, consider last 100 days UP-TO Yesterday.
                            AND DATE_SUB(temp_1.local_date, INTERVAL 1 DAY)
        # WHERE temp_1.batter = 116662
        GROUP BY temp_1.batter, temp_1.local_date
;


# Calculating batting average for each player on those Rolling 100 days:
DROP TEMPORARY TABLE IF EXISTS temp_batting_average;

CREATE TEMPORARY TABLE temp_batting_average
(batter mediumint, local_date date, rolling_batting_average float,
INDEX batter_idx (batter), INDEX date_idx (local_date)) ENGINE=MyISAM

	    SELECT
	            batter,
	            local_date,
	            ROUND( SUM(Hit)/SUM(atBat) , 3 ) AS rolling_batting_average
	    FROM temp_rolling_dates
	    # WHERE batter = 116662
	    GROUP BY batter, local_date
;


# Resultant table for Rolling Batting Average (Last 100 days):
DROP TABLE IF EXISTS player_rolling_average;

CREATE TABLE player_rolling_average
(batter mediumint, game_id smallint, rolling_batting_average float,
INDEX batter_idx (batter), INDEX date_idx (game_id)) ENGINE=MyISAM

        SELECT
                table_1.batter,
                table_1.game_id,
                table_1.local_date,
                table_2.rolling_batting_average
                # COUNT(*) as cnt
        FROM temp_player_game_details table_1
        JOIN temp_batting_average table_2
                ON table_1.local_date = table_2.local_date
        # WHERE table_1.batter = 116662
        ORDER BY table_1.game_id
;

# Results: Rolling Batting Average (Past 100 days):
SELECT * FROM player_rolling_average
LIMIT 0, 20;


# Results: Rolling Batting Average on a specific date, for all players for every game
SELECT * FROM player_rolling_average
WHERE local_date = '2010-08-18';