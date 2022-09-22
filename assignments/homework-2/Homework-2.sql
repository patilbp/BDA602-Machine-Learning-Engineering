# Check the table names for all tables in database
SHOW TABLES;

# Reference: https://en.wikipedia.org/wiki/Batting_average_(baseball)
# Batting average = Total hits / Total at-bats
# Reference: https://mariadb.com/kb/en/create-table/#create-or-replace

# Temporary table: I'm creating Temporary table to refer it at multiple places and increase my query speed.
# View for showing total hits and at bats for each player per game on specific date
DROP TEMPORARY TABLE IF EXISTS player_game_details;

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

# (1) Historic Batting Average for every player:

# I observed, few rows in table batter_counts had few 'atBat' values as '0'.
# So to eliminate the Division by '0' error, I added the condition of 'atBat' > 0.

DROP TABLE IF EXISTS player_historic_average;

CREATE TABLE player_historic_average
(batter mediumint, batting_average float,
INDEX batter_idx (batter)) ENGINE=MyISAM

        SELECT
                batter,
                ( SUM(Hit)/SUM(atBat) ) AS batting_average
        FROM batter_counts
        WHERE atBat > 0
        GROUP BY batter
;

# View the Historic Batting Average results:
SELECT * FROM player_historic_average;


# (2) Annual Batting Average for every player:
# I'm using my Temporary table here to create a resultant table from its results, for increasing query speed.

DROP TABLE IF EXISTS player_annual_average;

CREATE TABLE player_annual_average
(batter mediumint, local_date date, batting_average float,
INDEX batter_idx (batter), INDEX date_idx (local_date)) ENGINE=MyISAM

        SELECT
                batter,
                YEAR(local_date) AS year,
                SUM(Hit)/SUM(atBat) AS batting_average
        FROM temp_player_game_details
        GROUP BY batter, YEAR(local_date)
;

# View the Annual Batting Average results:
SELECT batter, year, batting_average FROM player_annual_average;


# (3) Rolling (Last 100 days) Batting Average for every player:
# I'm using Temporary table to calculate Rolling 100 days average, for increasing query speed.
# I also used the hint of self-join here.

DROP TABLE IF EXISTS player_rolling_average;

CREATE TABLE player_rolling_average
(rolling_average float, local_date date, batter mediumint, game_id smallint,
# atBat tinyint, Hit tinyint,
INDEX batter_idx (batter), INDEX game_idx (game_id), INDEX date_idx (local_date)) ENGINE=MyISAM

        SELECT
                ( SUM(temp_1.Hit) / SUM(temp_1.atBat) ) as rolling_average,
                temp_2.local_date,
                temp_2.batter,
                temp_2.game_id,
                COUNT(*) as cnt
        FROM temp_player_game_details temp_2
        JOIN temp_player_game_details temp_1
                ON temp_1.batter = temp_2.batter
                       AND temp_1.local_date > DATE_SUB (temp_2.local_date, INTERVAL 100 DAY)
                       AND temp_1.local_date < DATE_SUB(temp_2.local_date, INTERVAL 1 DAY)
        # Added below game_id to limit result for a particular game
        # WHERE temp_2.game_id = 10000
        GROUP BY temp_2.game_id ,temp_2.batter, temp_2.local_date
;

# View the Rolling (Past 100 days) Batting Average results for all players for every game:
SELECT * FROM player_rolling_average;

# View the Rolling (Past 100 days) Batting Average results for all players for every game on a Specific Date:
SELECT * FROM player_rolling_average
WHERE local_date = '2008-04-14';