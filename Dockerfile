FROM ubuntu

RUN mkdir /hw6_docker
RUN chmod a+x /hw6_docker
RUN chown 1000:1000 /hw6_docker

# Copy over code
COPY ./hw6_bash.sh /hw6_docker/hw6_bash.sh
COPY ./baseball.sql /hw6_docker/baseball.sql
COPY ./batting_average.sql /hw6_docker/batting_average.sql

# Run app
RUN chmod +x /hw6_docker/hw6_bash.sh

RUN apt-get update
RUN apt-get install -y mysql-client

CMD /hw6_docker/hw6_bash.sh