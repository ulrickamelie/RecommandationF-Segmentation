from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.sql.functions import isnan, when, count, col, round


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator 

import numpy as np

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
#from sklearn.cluster import KMeans
#from sklearn.cluster import KMeans
import numpy as np

from pyspark.sql.functions import col, explode
from pyspark.ml import Pipeline

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.classification import LogisticRegression

from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

def convert_data(df):
    stringIndexer = StringIndexer(inputCol="Gender", outputCol="GenderIndexed")
    model = stringIndexer.fit(df)
    model_gender = model.transform(df)
    model_gender = model_gender.drop('Gender','CustomerID')
    model_gender =  model_gender.select('Age','Annual Income (k$)','Spending Score (1-100)','GenderIndexed')
    return  model_gender
    

def convert_vector(df):
	columns = ['Annual Income (k$)','Spending Score (1-100)', 'Age']
	columns.append('GenderIndexed')
	vecAssembler = VectorAssembler(inputCols= columns,outputCol="features")
	new_v = vecAssembler.transform(df)
	return new_v
	
#Standardiser

#def stand_vector(df):
	#scaler = StandardScaler(inputCol='features', outputCol="scaledFeatures",withStd=True, withMean=True)
	#scalerModel = scaler.fit(df)
	#scaledData = scalerModel.transform(df)
	#return scaledData

def BestKmeanwithSil(df,nb_cluster):
	ks = np.arange(2, nb_cluster)
	max_value = 0
	silhouette_temp = 0
	for k in ks:
		kmeans = KMeans().setK(k).setSeed(1)
		model = kmeans.fit(df)
		predictions = model.transform(df)
		evaluator = ClusteringEvaluator()
		silhouette = evaluator.evaluate(predictions)
		#print("Silhouette with squared euclidean distance = " + str(silhouette) + " with %d" % (k))
		if (silhouette > silhouette_temp):
			max_value = k
			silhouette_temp = silhouette
		else :
			silhouette_temp = silhouette_temp
	return(max_value)
	
	
def kmeans_algo(df,nb_cluster):
    kmeans = KMeans().setK(nb_cluster).setSeed(1).setFeaturesCol("features")
    #kmeans.setMaxIter(300)
    model = kmeans.fit(df)
    #model.setPredictionCol("Cluster")
    df_predic = model.transform(df)
    #centers = model.clusterCenters()
    return df_predic
    
    
#PCA with kmeans

def df_for_pca(df):
	pca = PCA(2)
	#pred = df['prediction']
	df_2 = df.drop('prediction','Gender','features','scaledFeatures')
	df_2 = df_2.toPandas()
	df_2 = pca.fit_transform(df_2)
	return df_2
	
def PCA_kmeans(df,nb_cluster):
	from sklearn.cluster import KMeans

	#Initialize the class object
	kmeans = KMeans(n_clusters= nb_cluster)

	#predict the labels of clusters.
	label = kmeans.fit_predict(df)
	
	return label
	
	
#Recommandation

def rating_cast(df):
	#df3 = df1.join(df, ['movieId'], 'left')
	df = df.\
    withColumn('userId', col('userId').cast('integer')).\
    withColumn('movieId', col('movieId').cast('integer')).\
    withColumn('rating', col('rating').cast('float')).\
    drop('timestamp')
	return df
	
def ALS_recommandation(df,nb_film):
	(train, test) = df.randomSplit([0.8, 0.2], seed = 1234)
	als = ALS(maxIter=10, rank=50, regParam=0.15,userCol="userId", itemCol="movieId", ratingCol="rating", nonnegative = True, implicitPrefs = False, coldStartStrategy="drop")
	model = als.fit(train)
	prediction=model.transform(test)
	evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
	rmse=evaluator.evaluate(prediction)
	print("RMSE="+str(rmse))
	reco = model.recommendForAllUsers(nb_film)
	return reco


def recommandation_tab(df):
	df = df\
    .withColumn("rec_exp", explode("recommendations"))\
    .select('userId', col("rec_exp.movieId"), col("rec_exp.rating"))
	return df

def join_rating_movie(df,df1):
	df3=df.join(df1, on='movieId')
	return df3


### deuxième jeux de données Mall Cusomers

def transformeD(df):
	indexers = [StringIndexer(inputCol="Gender", outputCol="GenderIndexed") , 
            StringIndexer(inputCol="Branch", outputCol="BranchIndexed") ,
            StringIndexer(inputCol="City", outputCol="CityIndexed"),
            StringIndexer(inputCol="Customer type", outputCol="CustomerIndexed"),
            StringIndexer(inputCol="Payment", outputCol="PaymentIndexed")]

	pipeline = Pipeline(stages=indexers)
	df1 = pipeline.fit(df).transform(df)

	df1 = df1.drop('Invoice ID','Gender','CustomerID','Tax 5%','Date','Time','cogs','gross income','Rating','Branch','City','Customer type')
	df1 = df1.select('Product line','CustomerIndexed','GenderIndexed','PaymentIndexed','BranchIndexed','CityIndexed','Total')

	columns = ['BranchIndexed','CustomerIndexed','PaymentIndexed','CityIndexed']
	columns.append('GenderIndexed')
	vecAssembler = VectorAssembler(inputCols= columns,outputCol="features")
	new_v = vecAssembler.transform(df1)
	return new_v

def classificateur(df_prediction,train,test):
	#df_prediction = df_prediction.withColumnRenamed("prediction", "prediction_Kmeans")
	#train, test = df_prediction.randomSplit([0.7, 0.3], seed = 2018)
	rf = RandomForestClassifier(labelCol="prediction_Kmeans", featuresCol="features")
	model_tree = rf.fit(train)
	predictions_Random = model_tree.transform(test)
	predictions_Random.show()
	evaluator = MulticlassClassificationEvaluator(labelCol="prediction_Kmeans", predictionCol="prediction", metricName="accuracy")
	accuracy = evaluator.evaluate(predictions_Random)
	print("l'accuracy est de",accuracy)

def logicR(train,test):
	logistic = LogisticRegression(labelCol='GenderIndexed').fit(train)
	prediction = logistic.transform(test)
	TN = prediction.filter('prediction = 0 AND GenderIndexed = prediction').count()
	TP = prediction.filter('prediction = 1 AND GenderIndexed = prediction').count()
	FN = prediction.filter('prediction = 0 AND GenderIndexed = 1').count()
	FP = prediction.filter('prediction = 1 AND GenderIndexed = 0').count()

	# Calculate precision and recall
	precision = TP / (TP + FP)
	recall = TP / (TP + FN)
	print('precision = {:.2f}\nrecall   = {:.2f}'.format(precision, recall))

# Find weighted precision
	multi_evaluator = MulticlassClassificationEvaluator(labelCol='GenderIndexed')
	weighted_precision = multi_evaluator.evaluate(prediction, {multi_evaluator.metricName: "weightedPrecision"})

# Find AUC
	binary_evaluator = BinaryClassificationEvaluator(labelCol='GenderIndexed')
	auc = binary_evaluator.evaluate(prediction, {binary_evaluator.metricName: "areaUnderROC"})

	print('weighted_precision =',weighted_precision)
	print('Accuracy =',auc)











	
