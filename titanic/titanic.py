from pyspark import SparkConf, SparkContext
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import Bucketizer
import numpy as np
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import *
from math import log
from pyspark.ml.feature import ChiSqSelector

conf = SparkConf().setMaster("local").setAppName("Titanic")
sc = SparkContext(conf = conf)
sql = SQLContext(sc)

#function to parse a record
def mapParseLine(line):
    columns = line.split(',')
    PassengerId = columns[0]
    Survived = columns[1]
    Pclass = columns[2]
    LName = columns[3]
    Title = columns[4].split(".", 1)[0].strip()
    Sex = columns[5]
    Age = float(0 if columns[6] == '' else columns[6])
    SibSp = columns[7]
    Parch = columns[8]
    Fare = float(0 if columns[10] == '' else columns[10])
    Embarked = columns[12]
    return Row(passengerId = PassengerId, survived=Survived, title=Title, pclass=Pclass, sex=Sex, age=Age, sibsp=SibSp, parch=Parch, fare=float(Fare), embarked=Embarked)

def mapParseLineTest(line):
    columns = line.split(',')
    PassengerId = columns[0]
    Pclass = columns[1]
    LName = columns[2]
    Title = columns[3].split(".", 1)[0].strip()
    Sex = columns[4]
    Age = float(0 if columns[5] == '' else columns[6])
    SibSp = columns[6]
    Parch = columns[7]
    Fare = float(0 if columns[9] == '' else columns[9])
    Embarked = columns[11]
    return Row(passengerId = PassengerId, pclass=Pclass, title=Title, sex=Sex, age=Age, sibsp=SibSp, parch=Parch, fare=float(Fare), embarked=Embarked)

#function to plot distribution of generic variable

def plotGenericDistribution(train, variable):
    distinct_Pclass = sorted(train.select(variable, "survived").groupBy([variable]).count().withColumnRenamed("count", "counts").collect())
    distinct_Pclass_survived = sorted(train.select(variable, "survived").filter("survived='1'").groupBy([variable]).count().withColumnRenamed("count", "counts").collect())
    # defining subplots
    f, (total_plot, survival_plot, mean_plot) = plt.subplots(1,3)
    f.tight_layout()
    #plot distinct Pclass values - Total
    if(variable == 'pclass'):
        objects = [i.pclass for i in distinct_Pclass]
    if(variable == 'embarked'):
        objects = [i.embarked for i in distinct_Pclass]
    if(variable == 'sex'):
        objects = [i.sex for i in distinct_Pclass]
    if(variable == 'ageBucketed'):
        objects = [i.ageBucketed for i in distinct_Pclass]
    if(variable == 'fareBucketed'):
        objects = [i.fareBucketed for i in distinct_Pclass]
    y_pos = np.arange(len(objects))
    values = [i.counts for i in distinct_Pclass]
    total_plot.bar(y_pos, values, align='center', alpha=0.5)
    total_plot.set_xticks(y_pos)
    total_plot.set_xticklabels(objects)
    #total_plot.set_xlabel(objects)
    total_plot.set_ylabel('Count')
    total_plot.set_title(variable + ' Count Dist')
    #plot distinct Pclass values - Survived
    values1 = [i.counts for i in distinct_Pclass_survived]
    survival_plot.bar(y_pos, values1, align='center', alpha=0.5)
    survival_plot.set_xticks(y_pos)
    survival_plot.set_xticklabels(objects)
    survival_plot.set_ylabel('Survived = 1')
    survival_plot.set_title(variable + ' Survival Dist')
    #plot distinct Pclass values - Mean
    values_new = [1.0*int(b) / int(m) for b,m in zip(values1, values)]
    values_mean = [ '%.2f' % elem for elem in values_new ]
    mean_plot.bar(y_pos, values_mean, align='center', alpha=0.5)
    mean_plot.set_xticks(y_pos)
    mean_plot.set_xticklabels(objects)
    mean_plot.set_ylabel('Survived = 1 / Count')
    mean_plot.set_title(variable + ' Mean Dist')
    plt.show()
    

# load training and testing datasets
train_data = sc.textFile('D:\\temp\\rshah\\datasets\\titanic\\train.csv')
test_data = sc.textFile('D:\\temp\\rshah\\datasets\\titanic\\test.csv')

#remove header row
header_train = train_data.first()

train = train_data.filter(lambda row:row!=header_train).map(mapParseLine).toDF().cache()

#lets check the correlation of Each column Values with rate of survival
# 1. first lets start with Pclass
#plotGenericDistribution(train, 'pclass')
#from the graphs we saw that there is a decent correlation between PClass and Survival rate
#ppl from Pclass = 1 had better chances of survival ~ 65%, hence we should be using PClass as one of the features


# 2. Lets check Embarked distribution with rate of survival
# >>> train.select("embarked").groupBy("embarked").count().collect()
# [Row(embarked=u'Q', count=77), Row(embarked=u'C', count=168), Row(embarked=u'S', count=644), Row(embarked=u'', count=2)]
# Since there are couple of records with NULL embarked, lets replace those with 'S' (most frequent one)
train = train.replace('', 'S', 'embarked')
#plotGenericDistribution(train, 'embarked')
# we see that embarked = 'C' had better chance of survival

# 3. Lets check Gender distribution with Survival
#plotGenericDistribution(train, 'sex')
# Quite famously around, 75% of the female survived

# 4. Lets bucket age values and then plot its survival distribution
# replace NULL age values with mean value

mean_age = round(train.filter("age != 0").groupBy().mean('age').collect()[0][0], 2)
#train_df = train.replace(0, mean_age, 'age')
#train_df = train.filter("age != 0")
mean_fare = round(train.filter("fare != 0").groupBy().mean('fare').collect()[0][0], 2)
train_df = train.replace(0, mean_fare, 'fare')

age_split = [0.0, 13.0, 18.0, 23.0, 29.0, 34.0, 43.0, 150.0]
ageBucketizer = Bucketizer(splits=age_split, inputCol="age", outputCol="ageBucketed")
bucketed = ageBucketizer.transform(train_df)

#plotGenericDistribution(bucketed, 'ageBucketed')
# again quite famously kids < 18 years of age had the highest survival rate
#x = train_df.select("fare").collect()
#x_list = [i[0] for i in x]
#x_log = [log(i,10) for i in x_list]
#n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
#plt.show()
# 5. lets bucket Fare and try to plot its distribution

udf = UserDefinedFunction(lambda x: round(log(x,10), 5), DoubleType())

train_df = train_df.withColumn("fare_log", udf("fare")).drop("fare").withColumnRenamed("fare_log", "fare")
fare_split = [0.0, 0.9, 1.2, 1.5, 10.0]
fareBucketizer = Bucketizer(splits=fare_split, inputCol='fare', outputCol='fareBucketed')
bucketed = fareBucketizer.transform(train_df)

#plotGenericDistribution(bucketed, 'fareBucketed')

# a map of more aggregated titles
Title_Dictionary = {
                    "Capt":       1.0,
                    "Col":        1.0,
                    "Major":      1.0,
                    "Jonkheer":   2.0,
                    "Don":        2.0,
                    "Sir" :       2.0,
                    "Dr":         1.0,
                    "Rev":        1.0,
                    "the Countess":2.0,
                    "Dona":       2.0,
                    "Mme":        3.0,
                    "Mlle":       4.0,
                    "Ms":         3.0,
                    "Mr" :        5.0,
                    "Mrs" :       3.0,
                    "Miss" :      4.0,
                    "Master" :   6.0,
                    "Lady" :      2.0

                    }
    
udf2 = UserDefinedFunction(lambda x: (Title_Dictionary[x]), DoubleType())
train_df = train_df.withColumn("title_mapped", udf2("title")).drop("title").withColumnRenamed("title_mapped", "title")

# to begin lets try to train the model using these 4 variables
# Pclass, Embarked, Gender, Age
# index Embarked and Gender variables
# assemble all the features using VectorAssembler
# Train using Decision Tree Classifier
# Chain everything into a Pipeline
genderIndexer = StringIndexer(inputCol="sex", outputCol="genderIndexed")
ageBucketizer = Bucketizer(splits=age_split, inputCol="age", outputCol="ageBucketed")
embarkedIndexer = StringIndexer(inputCol="embarked", outputCol="embarkedIndexed")
pclassIndexer = StringIndexer(inputCol="pclass", outputCol="pclassIndexed")
featureAssembler = VectorAssembler(inputCols=["genderIndexed", "title", "pclassIndexed", "embarkedIndexed", "ageBucketed", "fareBucketed"], outputCol="features")
labelIndexer = StringIndexer(inputCol="survived", outputCol="label")

selector = ChiSqSelector(featuresCol="features", outputCol="selectedFeatures", labelCol="label")

dt = RandomForestClassifier(labelCol="label", featuresCol="selectedFeatures")
dt2 = DecisionTreeClassifier(labelCol="label", featuresCol="selectedFeatures")
pipeline = Pipeline(stages=[genderIndexer, ageBucketizer, fareBucketizer, embarkedIndexer, pclassIndexer, labelIndexer, featureAssembler, selector, dt])

#model = pipeline.fit(train_df)

#    .addGrid(dt.maxDepth, [3, 5, 7, 10]) \
#    .addGrid(dt.numTrees, [5, 10, 15, 20]) \
paramGrid = ParamGridBuilder() \
    .addGrid(selector.numTopFeatures, [2, 3,4]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=10)  # use 3+ folds in practice

# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(train_df)


#model = pipeline.fit(train_df)

header_test = test_data.first()
test_df = test_data.filter(lambda row:row!=header_test).map(mapParseLineTest).toDF().cache()
# replace NULL age values with mean value

#mean_age_test = round(test_df.filter("age != 0").groupBy().mean('age').collect()[0][0], 2)
#test_df = test_df.replace(0, mean_age, 'age')
#mean_fare = round(test_df.filter("fare != 0").groupBy().mean('fare').collect()[0][0], 2)
test_df = test_df.replace(0, mean_fare, 'fare')
test_df = test_df.withColumn("fare_log", udf("fare")).drop("fare").withColumnRenamed("fare_log", "fare")
test_df = test_df.replace('', 'S', 'embarked')
test_df = test_df.withColumn("title_mapped", udf2("title")).drop("title").withColumnRenamed("title_mapped", "title")

### predictions
pred = cvModel.transform(test_df)
pred = pred.withColumnRenamed('prediction', 'Survived')
pred = pred.select(pred.passengerId, pred.Survived.cast("int"))

print cvModel.selectedFeatures
outputCols= cvModel.getOutputCol()
for i in outputCols:
    print i
feature_cols =  cvModel.getFeaturesCol()
for i in feature_cols:
    print i

#write output to csv file
pred.write.csv("D:\\temp\\rshah\\spark-2.0.1-bin-hadoop2.7\\rshah_scripts\\titanic\\first_trial.csv")

print selector.getNumTopFeatures()
print selector.getOutputCol()
print selector.getFeaturesCol()
# next plan is to do PCA on fare and pclass: assuming higher fare meaning better pclass.
# still need to perform the correlation between both the values
