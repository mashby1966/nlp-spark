import sparknlp

# Start Spark NLP session
spark = sparknlp.start()

print("Spark NLP Version:", sparknlp.version())
print("Spark Session Version:", spark.version)

# Test Spark
data = spark.createDataFrame([("Hello NLP!",)], ["text"])
data.show()