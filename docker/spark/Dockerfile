FROM apache/spark:3.5.3-python3

USER root

# Pre-download MongoDB Spark Connector (Spark 3.5 / Scala 2.12) into Spark jars
ARG MONGO_CONNECTOR_VERSION=10.4.1
ARG MONGO_DRIVER_VERSION=5.2.1
RUN curl -fSL -o /opt/spark/jars/mongo-spark-connector_2.12-${MONGO_CONNECTOR_VERSION}.jar \
      "https://repo1.maven.org/maven2/org/mongodb/spark/mongo-spark-connector_2.12/${MONGO_CONNECTOR_VERSION}/mongo-spark-connector_2.12-${MONGO_CONNECTOR_VERSION}.jar" \
    && curl -fSL -o /opt/spark/jars/mongodb-driver-sync-${MONGO_DRIVER_VERSION}.jar \
      "https://repo1.maven.org/maven2/org/mongodb/mongodb-driver-sync/${MONGO_DRIVER_VERSION}/mongodb-driver-sync-${MONGO_DRIVER_VERSION}.jar" \
    && curl -fSL -o /opt/spark/jars/mongodb-driver-core-${MONGO_DRIVER_VERSION}.jar \
      "https://repo1.maven.org/maven2/org/mongodb/mongodb-driver-core/${MONGO_DRIVER_VERSION}/mongodb-driver-core-${MONGO_DRIVER_VERSION}.jar" \
    && curl -fSL -o /opt/spark/jars/bson-${MONGO_DRIVER_VERSION}.jar \
      "https://repo1.maven.org/maven2/org/mongodb/bson/${MONGO_DRIVER_VERSION}/bson-${MONGO_DRIVER_VERSION}.jar" \
    && curl -fSL -o /opt/spark/jars/bson-record-codec-${MONGO_DRIVER_VERSION}.jar \
      "https://repo1.maven.org/maven2/org/mongodb/bson-record-codec/${MONGO_DRIVER_VERSION}/bson-record-codec-${MONGO_DRIVER_VERSION}.jar" \
    && mkdir -p /opt/app /home/spark/.ivy2 \
    && chown -R spark:spark /opt/app /home/spark/.ivy2

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /opt/app
ENV PYTHONPATH=/opt/app
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

USER spark
