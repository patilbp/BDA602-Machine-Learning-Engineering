import sys

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from RollingAvgTransform import RollingBatAvgTransform

# mariadb-java-client-3.0.8.jar
# jdbc:mariadb://localhost:3306/baseball


def main():

    spark = SparkSession.builder.master("local[*]").getOrCreate()

    database = "baseball"
    port = "3306"
    user = "root"
    password = "eldudeH.22"  # pragma: allowlist secret

    bc = (
        spark.read.format("jdbc")
        .options(
            url=f"jdbc:mariadb://localhost:{port}/{database}",
            driver="org.mariadb.jdbc.Driver",
            dbtable="(Select bc.batter, \
                            bc.game_id, \
                            IF NULL(bc.Hit, 0), \
                            IF NULL(bc.atBat, 0), \
                            DATE(g.local_date) as local_dt \
                    FROM batter_counts bc \
                    JOIN game g \
                        ON bc.game_id = g.game_id \
                    )temp_player_game_details",
            user=user,
            password=password,
        )
        .load()
    )

    # Create temporary view to use further
    bc.createOrReplaceTempView("temp_player_game_details")
    bc.persist(StorageLevel.DISK_ONLY)

    # Temporary table details
    player_bc = spark.sql(
        "(SELECT * \
                            FROM temp_player_game_details )"
    )
    player_bc.show()

    # Creating Rolling Batting Average table
    # Note: Trying to do this calculation on Transformer side (in-progress)
    player_rolling_average = spark.sql(
        """
        (
            SELECT
                temp_1.batter,
                temp_1.game_id,
                temp_1.local_date,
                IF(temp_2.atBat = 0, 0, SUM(temp_2.Hit) / SUM(temp_2.atBat)) AS rolling_batting_average
            FROM temp_player_game_details temp_1
            JOIN temp_player_game_details temp_2
                ON  temp_1.batter = temp_2.batter
                        AND
                    temp_2.local_date
                        BETWEEN DATE_SUB(temp_1.local_date, INTERVAL 100 DAY)
                        AND DATE_SUB(temp_1.local_date, INTERVAL 1 DAY)
            GROUP BY temp_1.batter, temp_1.game_id, temp_1.local_date
            ORDER BY temp_1.batter, temp_1.game_id, temp_1.local_date;
        )
        """
    )

    # Calling out the Custom RDD Transformer
    rolling_avg = RollingBatAvgTransform(
        inputCols=["rolling_batting_average"], outputCol="rolling_average"
    )

    # Implementing the transformer on -> (rolling average) sql query
    # Reference: https://teaching.mrsharky.com/sdsu_fall_2020_lecture05.html#/7/7/3
    rolling_avg_results = rolling_avg.transform(player_rolling_average)

    # print("Rolling Batting Average (over last 100 days): \n")
    rolling_avg_results.show()


if __name__ == "__main__":
    sys.exit(main())
