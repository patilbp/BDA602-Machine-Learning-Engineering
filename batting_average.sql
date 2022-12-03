# BDA 602 : Assignment 6
# Rolling Batting Averages for all players in game_id = '12560'.

# Extras:
# (1) Variable Datatype: defined data types for temporary/non-temporary table.
#   - Reason being, 'int' takes more space than 'tinyint'.
# (2) Indexes: used indexing on all tables.
# (3) Linters: used 'sql fluff' as linter.
# (4) Engine: I've used MyISAM Engine for my code (as more space needed).


# Use the baseball datasource
USE baseball;

# Player-wise game details
DROP TABLE IF EXISTS player_game_details;
CREATE TABLE player_game_details
(
    batter     MEDIUMINT,
    game_id    MEDIUMINT,
    local_date DATE,
    atBat      TINYINT,
    Hit        TINYINT
)
    ENGINE=MyISAM
(
    SELECT
        bc.batter,
        bc.game_id,
        CAST(g.local_date AS DATE) AS local_date,
        IFNULL(bc.atBat, 0) AS atBat,
        IFNULL(bc.Hit, 0) AS Hit
    FROM batter_counts bc
    JOIN game g
        ON bc.game_id = g.game_id
    ORDER BY bc.batter, bc.game_id, local_date
);

# Creating Indexes
CREATE UNIQUE INDEX batter_game_date_idx ON player_game_details (batter, game_id, local_date);
CREATE INDEX batter_idx ON player_game_details (batter);
CREATE INDEX game_idx ON player_game_details (game_id);
CREATE INDEX date_idx ON player_game_details (local_date);


# Rolling Batting Average
DROP TABLE IF EXISTS player_rolling_average;
CREATE TABLE player_rolling_average
(
    batter                  MEDIUMINT,
    game_id                 MEDIUMINT,
    local_date              DATE,
    rolling_batting_average FLOAT
)
    ENGINE=MyISAM
(
    SELECT
        temp_1.batter,
        temp_1.game_id,
        temp_1.local_date,
        IF(temp_2.atBat = 0, 0, SUM(temp_2.Hit) / SUM(temp_2.atBat)) AS rolling_batting_average
        # COUNT(*) as count_rows
    FROM player_game_details temp_1
    JOIN player_game_details temp_2
       ON temp_1.batter = temp_2.batter
              AND
          temp_2.local_date
              BETWEEN DATE_SUB(temp_1.local_date, INTERVAL 100 DAY)
              # Hint: Exclude Today's date, consider only up to Yesterday.
              AND DATE_SUB(temp_1.local_date, INTERVAL 1 DAY)
    # Filter out results for game_id = '12560'
    WHERE temp_1.game_id = 12560
    GROUP BY temp_1.batter, temp_1.game_id, temp_1.local_date
    ORDER BY temp_1.batter, temp_1.game_id, temp_1.local_date
);

# Creating Indexes
CREATE UNIQUE INDEX batter_game_date_idx ON player_rolling_average (batter, game_id, local_date);
CREATE INDEX batter_idx ON player_rolling_average (batter);
CREATE INDEX game_idx ON player_rolling_average (game_id);
CREATE INDEX date_idx ON player_rolling_average (local_date);