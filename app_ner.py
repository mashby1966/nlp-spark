import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.sql import SparkSession

# Start Spark NLP
spark = sparknlp.start()

# Step 1: Prepare real-world text
data = spark.createDataFrame([["Barack Obama was born in Hawaii and was elected president of the United States."]]).toDF("text")

# Step 2: Build NLP Pipeline
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

ner = NerDLModel.pretrained("ner_dl", "en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["document", "token", "ner"]) \
    .setOutputCol("ner_chunk")

# Step 3: Assemble pipeline
pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    tokenizer,
    ner,
    ner_converter
])

# Step 4: Run the pipeline
model = pipeline.fit(data)
result = model.transform(data)

# Step 5: View results
result.selectExpr("explode(ner_chunk) as entity").select("entity.result", "entity.metadata").show(truncate=False)