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


# (1) Historic Batting Average:
# Few rows in table batter_counts had 'atBat' values as '0'.
# To eliminate divide by '0' error, added IF-ELSE condition.
DROP TABLE IF EXISTS player_historic_average;

CREATE TABLE player_historic_average
(
    batter                   MEDIUMINT,
    atBat                    SMALLINT,
    Hit                      SMALLINT,
    historic_batting_average FLOAT,
    INDEX batter_idx (batter)
)
    ENGINE=MEMORY
    SELECT
        batter,
        SUM(Hit) AS Hit,
        SUM( IFNULL(atBat, 0) ) AS atBat,
        IF (atBat = 0, 0, SUM(Hit)/SUM(atBat)) AS historic_batting_average
    FROM batter_counts
    GROUP BY batter
    ORDER BY batter;


# Total hits and at bats for each player:
DROP TEMPORARY TABLE IF EXISTS temp_player_game_details;

CREATE TEMPORARY TABLE temp_player_game_details
(
    batter     MEDIUMINT,
    game_id    MEDIUMINT,
    local_date DATE,
    atBat      TINYINT,
    Hit        TINYINT,
    INDEX batter_game_date_idx (batter, game_id, local_date),
    INDEX batter_idx (batter),
    INDEX game_idx (game_id),
    INDEX date_idx (local_date)
)
    ENGINE=MyISAM
    SELECT
        bc.batter,
        bc.game_id,
        DATE(g.local_date) AS local_date,
        IFNULL(bc.atBat, 0) AS atBat,
        bc.Hit
    FROM batter_counts bc
    JOIN game g ON bc.game_id = g.game_id
    ORDER BY batter, game_id, local_date;


# (2) Annual Batting Average:
DROP TABLE IF EXISTS player_annual_average;

CREATE TABLE player_annual_average
(
    batter                 MEDIUMINT,
    local_date             DATE,
    annual_batting_average FLOAT,
    UNIQUE INDEX batter_year_idx (batter, year),
    INDEX batter_idx (batter),
    INDEX year_idx (year)
)
    ENGINE=MEMORY
    SELECT
        batter,
        YEAR(local_date) AS year,
        IF (atBat = 0, 0, SUM(Hit)/SUM(atBat)) AS annual_batting_average
    FROM temp_player_game_details
    GROUP BY batter, year
    ORDER BY batter, year;


# (3) Rolling Batting Average:
DROP TABLE IF EXISTS player_rolling_average;

CREATE TABLE player_rolling_average
(
    batter                  MEDIUMINT,
    game_id                 MEDIUMINT,
    local_date              DATE,
    rolling_batting_average FLOAT,
    INDEX batter_date_idx (batter, local_date),
    INDEX batter_idx (batter),
    INDEX game_idx (game_id),
    INDEX date_idx (local_date)
)
    ENGINE=MyISAM
    SELECT
        temp_1.batter,
        temp_1.game_id,
        temp_1.local_date,
        IF (temp_2.atBat = 0, 0, SUM(temp_2.Hit) / SUM(temp_2.atBat)) AS rolling_batting_average
        # COUNT(*) as count_rows
    FROM temp_player_game_details temp_1
    JOIN temp_player_game_details temp_2
        ON temp_1.batter = temp_2.batter
               AND
           temp_2.local_date
               BETWEEN DATE_SUB(temp_1.local_date, INTERVAL 100 DAY)
               # Hint: Exclude Today's date, consider only up to Yesterday.
               AND DATE_SUB(temp_1.local_date, INTERVAL 1 DAY)
    GROUP BY temp_1.batter, temp_1.game_id, temp_1.local_date
    ORDER BY temp_1.batter, temp_1.game_id, temp_1.local_date;


# Results:
# Historic Batting Average
SELECT * FROM player_historic_average;

# Annual Batting Average
SELECT batter, year, annual_batting_average FROM player_annual_average;

# Rolling Batting Average
SELECT * FROM player_rolling_average;
