# BDA602 : Assignment 2
# Historic, Annual & Rolling - Batting Averages for all players.

# Extras:
# (1) Variable Datatype: defined data types for temporary/non-temporary table.
#   - Reason being, 'int' takes more space than 'tinyint'.
# (2) Temporary table: called and referenced for optimization.
# (3) Indexes: used indexing on all tables.
# (4) Linters: used 'sql fluff' as linter.
# (5) Engine: I've used MyISAM Engine for my code.


# References:
# (a) Batting Average: https://en.wikipedia.org/wiki/Batting_average_(baseball)
# (b) CREATE TABLE: https://mariadb.com/kb/en/create-table/#create-table-select
# (c) MyISAM Engine: https://mariadb.com/kb/en/show-engines/
# (d) SQL Fluff Linter: https://www.sqlfluff.com/


# (1) Historic Batting Average for every player:
# Few rows in table batter_counts had 'atBat' values as '0'.
# To eliminate divide by '0' error, added condition of 'atBat'>0.
DROP TABLE IF EXISTS player_historic_average;

CREATE TABLE player_historic_average
(batter MEDIUMINT, atBat SMALLINT, Hit SMALLINT, historic_batting_average FLOAT, INDEX batter_idx (batter)) ENGINE=MyISAM
    SELECT
        batter,
        SUM(Hit) AS Hit,
        SUM( IFNULL(atBat,0) ) AS atBat,
        ROUND( SUM(Hit)/SUM(atBat) , 3 ) AS historic_batting_average
    FROM batter_counts
    WHERE atBat > 0
    GROUP BY batter;

# Results: Historic Batting Average:
SELECT * FROM player_historic_average
LIMIT 0, 20;


# Total hits and at bats for each player:
DROP TEMPORARY TABLE IF EXISTS temp_player_game_details;

CREATE TEMPORARY TABLE temp_player_game_details
(batter MEDIUMINT, game_id SMALLINT, local_date DATE, atBat TINYINT, Hit TINYINT, INDEX batter_idx (batter), INDEX game_idx (game_id), INDEX date_idx (local_date)) ENGINE=MyISAM
    SELECT
        bc.batter,
        bc.game_id,
        DATE(g.local_date) AS local_date,
        bc.atBat,
        bc.Hit
    FROM batter_counts bc
    JOIN game g ON bc.game_id = g.game_id
    WHERE bc.atBat > 0;


# (2) Annual Batting Average for every player:
DROP TABLE IF EXISTS player_annual_average;

CREATE TABLE player_annual_average
(batter MEDIUMINT, local_date DATE, annual_batting_average FLOAT, INDEX batter_idx (batter), INDEX date_idx (local_date)) ENGINE=MyISAM
    SELECT
        batter,
        YEAR(local_date) AS year,
        ROUND( SUM(Hit)/SUM(atBat) , 3 ) AS annual_batting_average
    FROM temp_player_game_details
    GROUP BY batter, YEAR(local_date);

# Results: Annual Batting Average
SELECT batter, year, annual_batting_average FROM player_annual_average
LIMIT 0, 20;


# (3) Rolling Batting Average:
# Two parts of calculation:
# (a) Find Rolling 100 days date fields and its Batting Average.
# (b) Joining back date to batter-game table (hint-self-join).
DROP TEMPORARY TABLE IF EXISTS temp_rolling_dates;

CREATE TEMPORARY TABLE temp_rolling_dates
(batter MEDIUMINT, local_date DATE, atBat SMALLINT, Hit SMALLINT, INDEX batter_idx (batter), INDEX date_idx (local_date)) ENGINE=MyISAM
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
        # Hint: Exclude Today's date, consider only up to Yesterday.
        AND DATE_SUB(temp_1.local_date, INTERVAL 1 DAY)
    # WHERE temp_1.batter = 116662
    GROUP BY temp_1.batter, temp_1.local_date;


# Calculating batting average for each player on those Rolling 100 days:
DROP TEMPORARY TABLE IF EXISTS temp_batting_average;

CREATE TEMPORARY TABLE temp_batting_average
(batter MEDIUMINT, local_date DATE, rolling_batting_average FLOAT, INDEX batter_idx (batter), INDEX date_idx (local_date)) ENGINE=MyISAM
	SELECT
        batter,
        local_date,
        ROUND( SUM(Hit)/SUM(atBat) , 3 ) AS rolling_batting_average
	FROM temp_rolling_dates
	# WHERE batter = 116662
	GROUP BY batter, local_date;


# Rolling Batting Average:
DROP TABLE IF EXISTS player_rolling_average;

CREATE TABLE player_rolling_average
(batter MEDIUMINT, game_id SMALLINT, rolling_batting_average FLOAT, INDEX batter_idx (batter), INDEX date_idx (game_id)) ENGINE=MyISAM
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
    ORDER BY table_1.game_id;

# Results: Rolling Batting Average:
SELECT * FROM player_rolling_average
LIMIT 20;


# Rolling Batting Average on specific date:
SELECT * FROM player_rolling_average
WHERE local_date = '2010-08-18'
LIMIT 20;
