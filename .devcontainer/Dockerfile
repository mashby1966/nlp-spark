# Use the official OpenJDK 11 slim image as the base
FROM openjdk:11-jdk-slim

# Environment variables for versions and paths
ENV SPARK_VERSION=3.5.0
ENV HADOOP_VERSION=3
ENV SPARK_NLP_VERSION=5.1.3
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH
ENV PYSPARK_PYTHON=python3
# Ensure /usr/lib is in the library path so that the native lib can be found
ENV LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH

# Install required OS packages (including procps for ps command)
RUN apt-get update && \
    apt-get install -y curl python3 python3-pip procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3 as the default and install common Python packages
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    pip3 install --no-cache-dir numpy pandas matplotlib seaborn scikit-learn

# Install PySpark and Spark NLP
RUN pip3 install pyspark==${SPARK_VERSION} spark-nlp==${SPARK_NLP_VERSION}

# Download and install Apache Spark from the Apache Archive
RUN mkdir -p /opt && \
    curl -fsSL "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" -o /opt/spark.tgz && \
    tar -xzvf /opt/spark.tgz -C /opt && \
    rm /opt/spark.tgz && \
    mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} $SPARK_HOME

# Download the TensorFlow JNI native library (libjnitensorflow.so) and put it in /usr/lib
RUN curl -fsSL "https://storage.googleapis.com/tensorflow/libjnitensorflow.so" -o /usr/lib/libjnitensorflow.so

# Set the working directory
WORKDIR /workspace

# Default command
CMD ["bash"]