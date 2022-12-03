# BDA602 : Assignment 5
# Baseball - Extracting features using mariaDB.
# Query Load time: 59 secs, 230 ms

# Extras:
# (1) Used Temporary table: called and referenced for optimization.
# (2) Indexes: used indexing on all tables.
# (3) Linters: used 'sql fluff' as linter.
# (4) Engine: I've used MEMORY (less space) /MyISAM (more space) Engine for my code.

# References:
# (a) Baseball Statistics:
# https://sarahesult.medium.com/common-mlb-statistics-which-stats-determine-a-teams-win-percentage-a6e0a83aa07c
# (b) Baseball Wikipedia: https://en.wikipedia.org/wiki/Baseball_statistics
# (c) MyISAM Engine: https://mariadb.com/kb/en/show-engines/
# (d) SQL Fluff Linter: https://www.sqlfluff.com/


# Part (A): Prior Game Stats
# Player game and team details
DROP TEMPORARY TABLE IF EXISTS player_game_details;
CREATE TEMPORARY TABLE player_game_details ENGINE=MyISAM AS
(
    SELECT
        bc.batter,
        bc.game_id,
        CAST(g.local_date AS DATE) AS local_date,
        bc.team_id
    FROM batter_counts bc
    JOIN game g
        ON bc.game_id = g.game_id
    ORDER BY bc.batter, bc.game_id, local_date
);

# Creating Indexes
CREATE UNIQUE INDEX batter_game_date_team_idx ON player_game_details (batter, game_id, local_date, team_id);
CREATE UNIQUE INDEX batter_game_date_idx ON player_game_details (batter, game_id, local_date);
CREATE INDEX batter_idx ON player_game_details (batter);
CREATE INDEX game_idx ON player_game_details (game_id);
CREATE INDEX date_idx ON player_game_details (local_date);
CREATE INDEX team_idx ON player_game_details (team_id);


# Feature #(1): home_total_batters
DROP TEMPORARY TABLE IF EXISTS home_players;
CREATE TEMPORARY TABLE home_players ENGINE=MEMORY AS
(
    SELECT
        game_id,
        team_id,
        COUNT(DISTINCT batter) AS total_batters
    FROM player_game_details
    GROUP BY game_id, team_id
);

# Creating Indexes
CREATE UNIQUE INDEX game_team_idx ON home_players (game_id, team_id);
CREATE INDEX game_idx ON home_players (game_id);
CREATE INDEX team_idx ON home_players (team_id);


# Response Variable: home-team_wins
# Feature #(2): away_streak
DROP TEMPORARY TABLE IF EXISTS home_team_wins;
CREATE TEMPORARY TABLE home_team_wins ENGINE=MEMORY AS
(
    SELECT
        game_id,
        team_id AS home_team_id,
        opponent_id AS away_team_id,
        IF(win_lose = 'W', 1, 0) AS home_team_wins,
        away_streak,
        CAST(local_date AS DATE) AS local_date
    FROM team_results
    WHERE home_away = 'H'
    ORDER BY game_id, home_team_id, away_team_id
);

# Creating Indexes
CREATE UNIQUE INDEX game_home_away_idx ON home_team_wins (game_id, home_team_id, away_team_id);
CREATE UNIQUE INDEX game_home_idx ON home_team_wins (game_id, home_team_id);
CREATE INDEX game_idx ON home_team_wins (game_id);
CREATE INDEX home_idx ON home_team_wins (home_team_id);
CREATE INDEX away_idx ON home_team_wins (away_team_id);


# Feature #(3): home_next_game_days
DROP TEMPORARY TABLE IF EXISTS home_team;
CREATE TEMPORARY TABLE home_team ENGINE=MEMORY AS
(
    SELECT
        ts1.game_id,
        ts1.team_id AS home_team_id,
        DATEDIFF(CAST(ts2.local_date AS DATE), CAST(ts1.local_date AS DATE)) AS home_next_game_days
    FROM team_streak ts1
    JOIN team_streak ts2
        ON ts1.pre_game_id = ts2.game_id
    WHERE ts1.home_away = 'H'
    ORDER BY ts1.game_id, home_team_id
);

# Creating Indexes
CREATE INDEX game_home_idx ON home_team (game_id, home_team_id);
CREATE INDEX game_idx ON home_team (game_id);
CREATE INDEX home_idx ON home_team (home_team_id);


# Feature #(4): away_next_game_days
DROP TEMPORARY TABLE IF EXISTS away_team;
CREATE TEMPORARY TABLE away_team ENGINE=MEMORY AS
(
    SELECT
        ts1.game_id,
        ts1.team_id AS away_team_id,
        DATEDIFF(CAST(ts2.local_date AS DATE), CAST(ts1.local_date AS DATE)) AS away_next_game_days
    FROM team_streak ts1
    JOIN team_streak ts2
        ON ts1.pre_game_id = ts2.game_id
    WHERE ts1.home_away = 'A'
    ORDER BY ts1.game_id, away_team_id
);

# Creating Indexes
CREATE INDEX game_away_idx ON away_team (game_id, away_team_id);
CREATE INDEX game_idx ON away_team (game_id);
CREATE INDEX away_idx ON away_team (away_team_id);


# Feature #(5): plateAppearance (from previous game played)
DROP TEMPORARY TABLE IF EXISTS game_team_details;
CREATE TEMPORARY TABLE game_team_details ENGINE=MEMORY AS
(
    SELECT
        tc.game_id,
        tc.team_id,
        tn.prior_game_id,
        tc.atBat,
        tc.Hit,
        tc.plateApperance
    FROM team_game_prior_next tn
    JOIN team_batting_counts tc
        ON tn.team_id = tc.team_id AND tn.game_id = tc.game_id
    WHERE tc.homeTeam = 1
    ORDER BY tc.game_id, tc.team_id, tn.prior_game_id
);

# Creating Indexes
CREATE UNIQUE INDEX game_team_prior_idx ON game_team_details (game_id, team_id, prior_game_id);
CREATE UNIQUE INDEX game_team_idx ON game_team_details (game_id, team_id);
CREATE INDEX game_idx ON game_team_details (game_id);
CREATE INDEX team_idx ON game_team_details (team_id);
CREATE INDEX prior_idx ON game_team_details (prior_game_id);


# Feature #(6): prior_batting_average (from previous game played)
DROP TEMPORARY TABLE IF EXISTS batting_average;
CREATE TEMPORARY TABLE batting_average ENGINE=MEMORY AS
(
    SELECT
        t1.game_id,
        t1.team_id,
        t2.Hit AS prior_Hit,
        t2.atBat AS prior_atBat,
        t2.Hit / NULLIF(t2.atBat,0) AS prior_Batting_Average,
        t2.plateApperance
    FROM game_team_details t1
    JOIN game_team_details t2
        ON t1.prior_game_id = t2.game_id
    ORDER BY t1.game_id, t1.team_id
);

# Creating Indexes
CREATE UNIQUE INDEX game_team_idx ON batting_average (game_id, team_id);
CREATE INDEX game_idx ON batting_average (game_id);
CREATE INDEX team_idx ON batting_average (team_id);


# Part (B): Starting Pitcher Stats
# Pitcher details for all games
DROP TEMPORARY TABLE IF EXISTS pitcher_details;
CREATE TEMPORARY TABLE pitcher_details ENGINE=MyISAM AS
(
    SELECT
        g.game_id,
        g.local_date,
        p.pitcher,
        p.Hit AS hits,
        ((p.endingInning - p.startingInning) + 1) AS innings_pitched,
        p.Strikeout AS strikeouts,
        p.Home_run AS home_runs
    FROM pitcher_counts p
    JOIN game g
        On g.game_id = p.game_id
    ORDER BY g.game_id, g.local_date
);

# Creating Indexes
CREATE UNIQUE INDEX game_date_pitcher_idx ON pitcher_details (game_id, local_date, pitcher);
CREATE UNIQUE INDEX game_pitcher_idx ON pitcher_details (game_id, pitcher);
CREATE UNIQUE INDEX date_pitcher_idx ON pitcher_details (local_date, pitcher);
CREATE INDEX game_idx ON pitcher_details (game_id);
CREATE INDEX date_idx ON pitcher_details (local_date);
CREATE INDEX pitcher_idx ON pitcher_details (pitcher);


# Feature #(7): innings_pitched
# Feature #(8): games_pitched
# Feature #(9): strikeouts_per_9
# Feature #(10): hits_per_9
# Feature #(11): home_runs_per_9
# Calculate 100 days rolling stats for all pitchers in every game
DROP TEMPORARY TABLE IF EXISTS rolling_stats;
CREATE TEMPORARY TABLE rolling_stats ENGINE=MyISAM AS
(
    SELECT
        p1.pitcher,
        p1.game_id,
        p1.local_date,
        SUM(p2.innings_pitched) AS innings_pitched,
        COUNT(DISTINCT (p2.game_id)) AS games_pitched,
        ((SUM(p2.strikeouts) / SUM(p2.innings_pitched)) * 9) AS strikeouts_per_9,
        ((SUM(p2.hits) / SUM(p2.innings_pitched)) * 9) AS hits_per_9,
        ((SUM(p2.home_runs) / SUM(p2.innings_pitched)) * 9) AS home_runs_per_9
    FROM pitcher_details p1
    LEFT JOIN pitcher_details p2
        ON p1.pitcher = p2.pitcher
               AND
           p2.local_date
               BETWEEN DATE_SUB(p1.local_date, INTERVAL 100 DAY)
               AND p1.local_date - INTERVAL 1 DAY
    GROUP BY p1.pitcher, p1.game_id, p1.local_date
    ORDER BY p1.pitcher, p1.game_id
);

# Creating Indexes
CREATE UNIQUE INDEX game_date_pitcher_idx ON rolling_stats (game_id, local_date, pitcher);
CREATE UNIQUE INDEX game_pitcher_idx ON rolling_stats (game_id, pitcher);
CREATE UNIQUE INDEX date_pitcher_idx ON rolling_stats (local_date, pitcher);
CREATE INDEX game_idx ON rolling_stats (game_id);
CREATE INDEX date_idx ON rolling_stats (local_date);
CREATE INDEX pitcher_idx ON rolling_stats (pitcher);


# Create home pitcher rolling stats
DROP TEMPORARY TABLE IF EXISTS home_pitcher_rolling_stats;
CREATE TEMPORARY TABLE home_pitcher_rolling_stats ENGINE=MyISAM AS
(
    SELECT
        rs.*,
        pc.homeTeam,
        pc.startingPitcher
    FROM rolling_stats rs
    JOIN pitcher_counts pc
        ON rs.game_id = pc.game_id AND rs.pitcher = pc.pitcher
    ORDER BY game_id
);

# Creating Indexes
CREATE UNIQUE INDEX game_date_pitcher_idx ON home_pitcher_rolling_stats (game_id, local_date, pitcher);
CREATE UNIQUE INDEX game_pitcher_idx ON home_pitcher_rolling_stats (game_id, pitcher);
CREATE UNIQUE INDEX date_pitcher_idx ON home_pitcher_rolling_stats (local_date, pitcher);
CREATE INDEX game_idx ON home_pitcher_rolling_stats (game_id);
CREATE INDEX date_idx ON home_pitcher_rolling_stats (local_date);
CREATE INDEX pitcher_idx ON home_pitcher_rolling_stats (pitcher);


# Calculate home pitcher stats
DROP TEMPORARY TABLE IF EXISTS home_stats;
CREATE TEMPORARY TABLE home_stats ENGINE=MyISAM AS
(
    SELECT
        game_id,
        local_date,
        pitcher,
        innings_pitched,
        games_pitched,
        strikeouts_per_9,
        hits_per_9,
        home_runs_per_9
    FROM home_pitcher_rolling_stats
    WHERE homeTeam = 1 AND startingPitcher = 1
);

# Creating Indexes
CREATE INDEX home_game_idx ON home_stats (game_id);


# Create away pitcher rolling stats
DROP TEMPORARY TABLE IF EXISTS away_pitcher_rolling_stats;
CREATE TEMPORARY TABLE away_pitcher_rolling_stats ENGINE=MyISAM AS
(
    SELECT
        rs.*,
        pc.awayTeam,
        pc.startingPitcher
    FROM rolling_stats rs
    JOIN pitcher_counts pc
        ON rs.game_id = pc.game_id AND rs.pitcher = pc.pitcher
    ORDER BY game_id
);

# Creating Indexes
CREATE UNIQUE INDEX game_date_pitcher_idx ON away_pitcher_rolling_stats (game_id, local_date, pitcher);
CREATE UNIQUE INDEX game_pitcher_idx ON away_pitcher_rolling_stats (game_id, pitcher);
CREATE UNIQUE INDEX date_pitcher_idx ON away_pitcher_rolling_stats (local_date, pitcher);
CREATE INDEX game_idx ON away_pitcher_rolling_stats (game_id);
CREATE INDEX date_idx ON away_pitcher_rolling_stats (local_date);
CREATE INDEX pitcher_idx ON away_pitcher_rolling_stats (pitcher);


# Calculate away pitcher stats
DROP TEMPORARY TABLE IF EXISTS away_stats;
CREATE TEMPORARY TABLE away_stats engine=MyISAM AS
(
    SELECT
        game_id,
        local_date,
        pitcher,
        innings_pitched,
        games_pitched,
        strikeouts_per_9,
        hits_per_9,
        home_runs_per_9
    FROM away_pitcher_rolling_stats
    WHERE awayTeam = 1 AND startingPitcher = 1
);

# Creating Indexes
CREATE INDEX away_game_idx ON away_stats (game_id);


# Feature #(12): innings_pitched_ratio
# Feature #(13): games_pitched_difference
# Feature #(14): strikeouts_per_9_difference
# Feature #(15): hits_per_9_difference
# Feature #(16): home_runs_per_9_difference

# Merge the home team stats and away team stats tables
DROP TEMPORARY TABLE IF EXISTS home_away_stats;
CREATE TEMPORARY TABLE home_away_stats ENGINE=MyISAM AS
(
    SELECT
        h.game_id,
        h.local_date,
        h.pitcher AS home_pitcher,
        a.pitcher AS away_pitcher,
        h.innings_pitched,
        h.games_pitched,
        h.strikeouts_per_9,
        h.hits_per_9,
        h.home_runs_per_9,
        (h.innings_pitched / a.innings_pitched) AS innings_pitched_ratio,
        (h.games_pitched - a.games_pitched) AS games_pitched_difference,
        (h.strikeouts_per_9 - a.strikeouts_per_9) AS strikeouts_per_9_difference,
        (h.hits_per_9 - a.hits_per_9) AS hits_per_9_difference,
        (h.home_runs_per_9 - a.home_runs_per_9) AS home_runs_per_9_difference
    FROM home_stats h
    JOIN away_stats a ON h.game_id = a.game_id
    ORDER BY h.game_id, h.local_date
);

# Creating Indexes
CREATE UNIQUE INDEX game_date_home_away_idx ON home_away_stats (game_id, local_date, home_pitcher, away_pitcher);
CREATE UNIQUE INDEX game_home_away_idx ON home_away_stats (game_id, home_pitcher, away_pitcher);
CREATE UNIQUE INDEX date_home_away_idx ON home_away_stats (local_date, home_pitcher, away_pitcher);
CREATE INDEX game_idx ON home_away_stats (game_id);
CREATE INDEX date_idx ON home_away_stats (local_date);
CREATE INDEX home_idx ON home_away_stats (home_pitcher);
CREATE INDEX away_idx ON home_away_stats (away_pitcher);


# Final Resultant table with all extracted features
DROP TABLE IF EXISTS pitcher_stats;
CREATE TABLE pitcher_stats ENGINE=MEMORY AS
(
    SELECT
        DISTINCT t1.game_id,
        t1.home_team_id,
        t1.away_team_id,
        t1.local_date,
        t1.home_team_wins,
        t1.away_streak,
        t2.home_next_game_days,
        t3.away_next_game_days,
        t4.prior_Hit AS home_prior_Hit,
        t4.prior_atBat AS home_prior_atBat,
        t4.prior_Batting_Average AS home_prior_Batting_Average,
        t4.plateApperance AS home_prior_plateApperance,
        t5.total_batters AS home_total_batters,
        t6.home_pitcher,
        t6.away_pitcher,
        t6.innings_pitched,
        t6.games_pitched,
        t6.strikeouts_per_9,
        t6.hits_per_9,
        t6.home_runs_per_9,
        t6.innings_pitched_ratio,
        t6.games_pitched_difference,
        t6.strikeouts_per_9_difference,
        t6.hits_per_9_difference,
        t6.home_runs_per_9_difference
    FROM home_team_wins t1
    JOIN home_team t2
        ON t1.game_id = t2.game_id AND t1.home_team_id = t2.home_team_id
    JOIN away_team t3
        ON t1.game_id = t3.game_id AND t1.away_team_id = t3.away_team_id
    JOIN batting_average t4
        ON t1.game_id = t4.game_id AND t1.home_team_id = t4.team_id
    JOIN home_players t5
        ON t1.game_id = t5.game_id AND t1.home_team_id = t5.team_id
    JOIN home_away_stats t6
        ON t1.game_id = t6.game_id
    ORDER BY t1.game_id, t1.home_team_id, t1.away_team_id, t1.local_date
);

# Creating Indexes
CREATE UNIQUE INDEX ps_game_home_away_date_idx ON pitcher_stats (game_id, home_team_id, away_team_id, local_date);
CREATE UNIQUE INDEX ps_game_home_away_idx ON pitcher_stats (game_id, home_team_id, away_team_id);
CREATE UNIQUE INDEX ps_game_home_date_idx ON pitcher_stats (game_id, home_team_id, local_date);
CREATE UNIQUE INDEX ps_game_away_date_idx ON pitcher_stats (game_id, away_team_id, local_date);
CREATE INDEX ps_game_idx ON pitcher_stats (game_id);
CREATE INDEX ps_home_idx ON pitcher_stats (home_team_id);
CREATE INDEX ps_away_idx ON pitcher_stats (away_team_id);
CREATE INDEX ps_date_idx ON pitcher_stats (local_date);


-- Show the final Table
SELECT * FROM pitcher_stats
LIMIT 20;