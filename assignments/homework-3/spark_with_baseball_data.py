import sys

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from rolling_bat_avg_transform import RollingBatAvgTransform

# mariadb-java-client-3.0.8.jar
# jdbc:mariadb://localhost:3306/baseball


def main():
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    database = "baseball"
    port = "3306"
    user = "root"
    password = "eldudeH.22"  # pragma: allowlist secret

    baseball_data = (
        spark.read.format("jdbc")
        .options(
            url=f"jdbc:mariadb://localhost:{port}/{database}",
            driver="org.mariadb.jdbc.Driver",
            dbtable="(SELECT bc.batter, \
                            bc.game_id, \
                            bc.Hit, \
                            bc.atBat, \
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
    baseball_data.createOrReplaceTempView("temp_player_game_details")
    baseball_data.persist(StorageLevel.DISK_ONLY)

    # Temporary table details
    player_bc = spark.sql(" SELECT * FROM temp_player_game_details ")
    player_bc.show()

    # Calling out the Custom RDD Transformer
    rolling_avg = RollingBatAvgTransform(
        inputCols=["bc.Hit", "bc.atBat"], outputCol="rolling_average"
    )

    # Implementing the transformer on -> (rolling average) sql query
    # Reference: https://teaching.mrsharky.com/sdsu_fall_2020_lecture05.html#/7/7/3
    rolling_avg_results = rolling_avg.transform(baseball_data)

    # print("Rolling Batting Average (over last 100 days): \n")
    rolling_avg_results.show()


if __name__ == "__main__":
    sys.exit(main())
