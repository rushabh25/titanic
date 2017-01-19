from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.sql import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import Bucketizer

sql = SQLContext(sc)

# load training and testing datasets
train_data = sc.textFile('D:\\temp\\rshah\\datasets\\titanic\\train.csv')
test_data = sc.textFile('D:\\temp\\rshah\\datasets\\titanic\\test.csv')

#remove header row
header_train = train_data.first()
header_test = test_data.first()

train = train_data.filter(lambda row:row!=header_train)
test = test_data.filter(lambda row:row!=header_test)

#get the required cols
cols_train = train.map(lambda x:x.split(',')).map(lambda x: Row(PassengerId = x[0], Survived=x[1], Gender=x[5], Age=x[6]))
cols_test = test.map(lambda x:x.split(',')).map(lambda x: Row(PassengerId= x[0], Gender=x[4], Age=x[5]))

#create dataframe to load into ML
train_df = sql.createDataFrame(cols_train)
test_df = sql.createDataFrame(cols_test)

#cast Age column to double
train_df = train_df.withColumn('Age2', train_df.Age.cast("double"))
train_df = train_df.drop('Age').withColumnRenamed('Age2', 'Age')

test_df = test_df.withColumn('Age2', test_df.Age.cast("double"))
test_df = test_df.drop('Age').withColumnRenamed('Age2', 'Age')

#fill null age values with average values
mean_value_train =train_df.groupBy().avg('Age').collect()[0][0]
mean_value_test =test_df.groupBy().avg('Age').collect()[0][0]

train_df = train_df.fillna(mean_value_train, subset=["Age"])
test_df = test_df.fillna(mean_value_test, subset=["Age"])

age_split = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 150.0]

#index gender, create a vector from gender column and index the label
genderIndexer = StringIndexer(inputCol="Gender", outputCol="GenderIndexed")
ageBucketizer = Bucketizer(splits=age_split, inputCol="Age", outputCol="AgeBucketed")
featureAssembler = VectorAssembler(inputCols=["GenderIndexed", "AgeBucketed"], outputCol="features")
labelIndexer = StringIndexer(inputCol="Survived", outputCol="label")

dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

#chain everything into a pipeline
pipeline = Pipeline(stages=[genderIndexer, ageBucketizer, labelIndexer, featureAssembler, dt])

#modeling begins
model = pipeline.fit(train_df)

# predictions
pred = model.transform(test_df)
pred = pred.withColumnRenamed('prediction', 'Survived')
pred = pred.select(pred.PassengerId, pred.Survived.cast("int"))

#write output to csv file
pred.write.csv("D:\\temp\\rshah\\datasets\\titanic\\first_trial.csv")
