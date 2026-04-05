from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import avg

# Create Spark Session
spark = SparkSession.builder.appName("InternetDataUsage").getOrCreate()

# Load dataset
data = spark.read.csv("data/monthly_internet_usage_dataset.csv", header=True, inferSchema=True)

# Feature selection
feature_columns = [
    "month_index",
    "avg_hourly_MB",
    "peak_hour_usage_MB",
    "offpeak_hour_usage_MB",
    "prev_month_usage_MB"
]

label_column = "total_monthly_MB"

# Assemble features
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data_assembled = assembler.transform(data)

# Train-test split
train, test = data_assembled.randomSplit([0.8, 0.2], seed=42)

# Train model
lr = LinearRegression(featuresCol="features", labelCol=label_column)
model = lr.fit(train)

# Predictions
predictions = model.transform(test)
predictions.select("user_id", "month", label_column, "prediction").show(10)

# Evaluation
evaluator = RegressionEvaluator(labelCol=label_column, predictionCol="prediction")

print("MAE:", evaluator.setMetricName("mae").evaluate(predictions))
print("RMSE:", evaluator.setMetricName("rmse").evaluate(predictions))
print("R2:", evaluator.setMetricName("r2").evaluate(predictions))

# Trend Analysis
monthly_trend = data.groupBy("year", "month") \
    .agg(avg("total_monthly_MB").alias("avg_usage_MB")) \
    .orderBy("year", "month")

monthly_trend.show()
