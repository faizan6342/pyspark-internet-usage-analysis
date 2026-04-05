from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import avg
import matplotlib.pyplot as plt

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
print("\nSample Predictions:\n")
predictions.select("user_id", "month", label_column, "prediction").show(10)

# Evaluation
evaluator = RegressionEvaluator(labelCol=label_column, predictionCol="prediction")

print("\nEvaluation Metrics:")
print("MAE:", evaluator.setMetricName("mae").evaluate(predictions))
print("RMSE:", evaluator.setMetricName("rmse").evaluate(predictions))
print("R2:", evaluator.setMetricName("r2").evaluate(predictions))

# Trend Analysis
monthly_trend = data.groupBy("year", "month") \
    .agg(avg("total_monthly_MB").alias("avg_usage_MB")) \
    .orderBy("year", "month")

print("\nMonthly Internet Usage Trend:")
monthly_trend.show()


# Visualization
# =========================

# Convert Spark DataFrame to Pandas
trend_pd = monthly_trend.toPandas()

# Sort values
trend_pd = trend_pd.sort_values(by=["year", "month"])

# Plot
plt.figure()
plt.plot(trend_pd["month"], trend_pd["avg_usage_MB"], marker='o')

plt.title("Monthly Internet Usage Trend (2023)")
plt.xlabel("Month")
plt.ylabel("Average Usage (MB)")
plt.xticks(range(1, 13))
plt.grid()
plt.tight_layout()

plt.show()

print("\nVisualization generated successfully!")

# Actual vs Predicted Graph
# =========================

pred_pd = predictions.select("total_monthly_MB", "prediction").toPandas()

plt.figure()
plt.scatter(pred_pd["total_monthly_MB"], pred_pd["prediction"])

plt.title("Actual vs Predicted Usage")
plt.xlabel("Actual Usage (MB)")
plt.ylabel("Predicted Usage (MB)")
plt.grid()
plt.show()


# Peak vs Off-Peak Usage
# =========================

peak_data = data.groupBy("month") \
    .agg(
        avg("peak_hour_usage_MB").alias("peak"),
        avg("offpeak_hour_usage_MB").alias("offpeak")
    ).orderBy("month")

peak_pd = peak_data.toPandas()

plt.figure()
plt.plot(peak_pd["month"], peak_pd["peak"], label="Peak Usage")
plt.plot(peak_pd["month"], peak_pd["offpeak"], label="Off-Peak Usage")

plt.legend()
plt.title("Peak vs Off-Peak Usage")
plt.xlabel("Month")
plt.ylabel("Usage (MB)")
plt.grid()
plt.show()

# Stop Spark session
spark.stop()
