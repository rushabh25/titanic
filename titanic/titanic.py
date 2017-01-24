from pyspark import SparkConf, SparkContext
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.sql import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import Bucketizer
import numpy as np
import matplotlib.pyplot as plt

conf = SparkConf().setMaster("local").setAppName("Titanic")
sc = SparkContext(conf = conf)
sql = SQLContext(sc)

#function to parse a record
def mapParseLine(line):
    columns = line.split(',')
    Survived = columns[1]
    Pclass = columns[2]
    LName = columns[3]
    FName = columns[4]
    Sex = columns[5]
    Age = columns[6]
    SibSp = columns[7]
    Parch = columns[8]
    Fare = columns[10]
    Embarked = columns[12]
    return Row(survived=Survived, pclass=Pclass, sex=Sex, age=Age, sibsp=SibSp, parch=Parch, fare=Fare, embarked=Embarked)

#function to plot distribution of PClass variable
def plotPClassDistribution(train):
    distinct_Pclass = sorted(train.select("pclass", "survived").groupBy(["pclass"]).count().withColumnRenamed("count", "counts").collect())
    distinct_Pclass_survived = sorted(train.select("pclass", "survived").filter("survived='1'").groupBy(["pclass"]).count().withColumnRenamed("count", "counts").collect())

    # defining subplots
    f, (total_plot, survival_plot, mean_plot) = plt.subplots(1,3)
    f.tight_layout()
    #plot distinct Pclass values - Total
    objects = [i.pclass for i in distinct_Pclass]
    y_pos = np.arange(len(objects))
    values = [i.counts for i in distinct_Pclass]
    total_plot.bar(y_pos, values, align='center', alpha=0.5)
    total_plot.set_xticks(y_pos)
    total_plot.set_xticklabels(objects)
    #total_plot.set_xlabel(objects)
    total_plot.set_ylabel('Count')
    total_plot.set_title('PClass Count Dist')

    #plot distinct Pclass values - Survived
    values1 = [i.counts for i in distinct_Pclass_survived]
    survival_plot.bar(y_pos, values1, align='center', alpha=0.5)
    survival_plot.set_xticks(y_pos)
    survival_plot.set_xticklabels(objects)
    survival_plot.set_ylabel('Survived = 1')
    survival_plot.set_title('PClass Survival Dist')

    #plot distinct Pclass values - Mean
    values_new = [1.0*int(b) / int(m) for b,m in zip(values1, values)]
    values_mean = [ '%.2f' % elem for elem in values_new ]
    mean_plot.bar(y_pos, values_mean, align='center', alpha=0.5)
    mean_plot.set_xticks(y_pos)
    mean_plot.set_xticklabels(objects)
    mean_plot.set_ylabel('Survived = 1 / Count')
    mean_plot.set_title('PClass Mean Dist')

    plt.show()
    plt.close(f)


#function to plot distribution of Embarked variable
def plotEmbarkedDistribution(train):
    distinct_Pclass = sorted(train.select("embarked", "survived").groupBy(["embarked"]).count().withColumnRenamed("count", "counts").collect())
    distinct_Pclass_survived = sorted(train.select("embarked", "survived").filter("survived='1'").groupBy(["embarked"]).count().withColumnRenamed("count", "counts").collect())

    # defining subplots
    f, (total_plot, survival_plot, mean_plot) = plt.subplots(1,3)
    f.tight_layout()
    #plot distinct Pclass values - Total
    objects = [i.embarked for i in distinct_Pclass]
    y_pos = np.arange(len(objects))
    values = [i.counts for i in distinct_Pclass]
    total_plot.bar(y_pos, values, align='center', alpha=0.5)
    total_plot.set_xticks(y_pos)
    total_plot.set_xticklabels(objects)
    #total_plot.set_xlabel(objects)
    total_plot.set_ylabel('Count')
    total_plot.set_title('Embarked Count Dist')

    #plot distinct Pclass values - Survived
    values1 = [i.counts for i in distinct_Pclass_survived]
    survival_plot.bar(y_pos, values1, align='center', alpha=0.5)
    survival_plot.set_xticks(y_pos)
    survival_plot.set_xticklabels(objects)
    survival_plot.set_ylabel('Survived = 1')
    survival_plot.set_title('Embarked Survival Dist')

    #plot distinct Pclass values - Mean
    values_new = [1.0*int(b) / int(m) for b,m in zip(values1, values)]
    values_mean = [ '%.2f' % elem for elem in values_new ]
    mean_plot.bar(y_pos, values_mean, align='center', alpha=0.5)
    mean_plot.set_xticks(y_pos)
    mean_plot.set_xticklabels(objects)
    mean_plot.set_ylabel('Survived = 1 / Count')
    mean_plot.set_title('Embarked Mean Dist')

    plt.show()
    plt.close(f)

# load training and testing datasets
train_data = sc.textFile('D:\\temp\\rshah\\datasets\\titanic\\train.csv')
test_data = sc.textFile('D:\\temp\\rshah\\datasets\\titanic\\test.csv')

#remove header row
header_train = train_data.first()

train = train_data.filter(lambda row:row!=header_train).map(mapParseLine).toDF().cache()

#lets check the correlation of Each column Values with rate of survival
# 1. first lets start with Pclass
plotPClassDistribution(train)
#from the graphs we saw that there is a decent correlation between PClass and Survival rate
#ppl from Pclass = 1 had better chances of survival ~ 65%, hence we should be using PClass as one of the features


# 2. Lets check Embarked distribution with rate of survival
# >>> train.select("embarked").groupBy("embarked").count().collect()
# [Row(embarked=u'Q', count=77), Row(embarked=u'C', count=168), Row(embarked=u'S', count=644), Row(embarked=u'', count=2)]
# Since there are couple of records with NULL embarked, lets replace those with 'S' (most frequent one)
embarked = train.replace('', 'S', 'embarked')
plotEmbarkedDistribution(embarked)


###get the required cols
##cols_train = train.map(lambda x:x.split(',')).map(lambda x: Row(PassengerId = x[0], Survived=x[1], Gender=x[5], Age=x[6]))
##cols_test = test.map(lambda x:x.split(',')).map(lambda x: Row(PassengerId= x[0], Gender=x[4], Age=x[5]))
##
###create dataframe to load into ML
##train_df = sql.createDataFrame(cols_train)
##test_df = sql.createDataFrame(cols_test)
##
###cast Age column to double
##train_df = train_df.withColumn('Age2', train_df.Age.cast("double"))
##train_df = train_df.drop('Age').withColumnRenamed('Age2', 'Age')
##
##test_df = test_df.withColumn('Age2', test_df.Age.cast("double"))
##test_df = test_df.drop('Age').withColumnRenamed('Age2', 'Age')
##
###fill null age values with average values
##mean_value_train =train_df.groupBy().avg('Age').collect()[0][0]
##mean_value_test =test_df.groupBy().avg('Age').collect()[0][0]
##
##train_df = train_df.fillna(mean_value_train, subset=["Age"])
##test_df = test_df.fillna(mean_value_test, subset=["Age"])
##
##age_split = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 150.0]
##
###index gender, create a vector from gender column and index the label
##genderIndexer = StringIndexer(inputCol="Gender", outputCol="GenderIndexed")
##ageBucketizer = Bucketizer(splits=age_split, inputCol="Age", outputCol="AgeBucketed")
##featureAssembler = VectorAssembler(inputCols=["GenderIndexed", "AgeBucketed"], outputCol="features")
##labelIndexer = StringIndexer(inputCol="Survived", outputCol="label")
##
##dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
##
###chain everything into a pipeline
##pipeline = Pipeline(stages=[genderIndexer, ageBucketizer, labelIndexer, featureAssembler, dt])
##
###modeling begins
##model = pipeline.fit(train_df)
##
### predictions
##pred = model.transform(test_df)
##pred = pred.withColumnRenamed('prediction', 'Survived')
##pred = pred.select(pred.PassengerId, pred.Survived.cast("int"))
##
###write output to csv file
##pred.write.csv("D:\\temp\\rshah\\datasets\\titanic\\first_trial.csv")
