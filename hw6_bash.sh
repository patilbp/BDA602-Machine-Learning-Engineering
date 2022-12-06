#!/bin/sh
# BDA 602 - HW5 : Bash script to check if Baseball data is available.
# If not available then add it.
sleep 10

DATA=`mysqlshow -h mariadb -peldudeH.22 -u root baseball`

# Check if the database exists or not
if ! mysql -h mariadb -uroot -peldudeH.22 -e 'use baseball'; then
  echo "Baseball DOES NOT exists"
    mysql -h mariadb -peldudeH.22 -u root -e "CREATE DATABASE IF NOT EXISTS baseball"
    mysql -h mariadb -peldudeH.22 -u root baseball < /hw6_docker/baseball.sql
else
  echo "Baseball Exists"
fi

# Add SQL file for calculating rolling batting average of game_id = '12560'
echo "Calling batting_average.sql file"
  mysql -h mariadb -peldudeH.22 -u root baseball < /hw6_docker/batting_average.sql

  mysql -h mariadb -u root -peldudeH.22 -e '
    USE baseball;
    SELECT * FROM player_rolling_average;' > /result/result.txt
