import numpy as np
import pandas as pd
from pandas import plotting
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib.colors import ListedColormap
from pylab import rcParams
from pyspark.sql.functions import col, explode


def gender_visu(df):
	df_visu = df.toPandas()
	sns.countplot(df_visu.Gender)
	plt.title('Nombre par sexe')
	plt.show()

def age_visu(df):
	df_visu = df.toPandas()
	sns.distplot(df_visu['Age'])
	plt.show()
	
def annual_visu(df):
	df_visu = df.toPandas()
	sns.distplot(df_visu['Annual Income (k$)'])
	plt.show()

def spending_visu(df):
	df_visu = df.toPandas()
	sns.distplot(df_visu['Spending Score (1-100)'])
	plt.show()	
	
def all_visu(df):
	df_visu = df.toPandas()
	sns.pairplot(df_visu[['Age','Annual Income (k$)','Spending Score (1-100)']])
	plt.show()	
	
	
def double_visu_MaleFem(df):	
	df_visu = df.toPandas()
	plt.figure(figsize=(25,5))
	plt.subplot(1,3,1)
	for gender in ['Male' , 'Female']:
		plt.scatter(x = 'Age' , y = 'Annual Income (k$)' , data = df_visu[df_visu['Gender'] == gender] ,s= 100, label = gender)
	plt.title('Age vs Annual Income ')
	plt.subplot(1,3,2)
	for gender in ['Male' , 'Female']:
		plt.scatter(x = 'Age',y = 'Spending Score (1-100)' ,data = df_visu[df_visu['Gender'] == gender] ,s= 100 , label = gender)
	plt.title('Age vs Spending')
	plt.legend()
	plt.show()


def AgeAndIncome(df):
	df_visu = df.toPandas()
	classes = ['cluster_1', 'cluster_2','cluster_3','cluster_4','cluster_5']
	colors = ListedColormap(['red', 'blue', 'purple','black','orange'])
	scatter = plt.scatter(x=df_visu['Age'],y=df_visu['Annual Income (k$)'],c=df_visu['prediction'],cmap=colors)
	plt.legend(handles=scatter.legend_elements()[0], labels=classes)


def IncomeAndScore(df):
	df_visu = df.toPandas()
	classes = ['cluster_1', 'cluster_2','cluster_3','cluster_4','cluster_5']
	colors = ListedColormap(['red', 'blue', 'purple','black','orange'])
	scatter =plt.scatter(x=df_visu['Annual Income (k$)'],y=df_visu['Spending Score (1-100)'],c=df_visu['prediction'],cmap=colors)
	plt.legend(handles=scatter.legend_elements()[0], labels=classes)
	
	
def AgeAndScore(df):
	df_visu = df.toPandas()
	classes = ['cluster_1', 'cluster_2','cluster_3','cluster_4','cluster_5']
	colors = ListedColormap(['red', 'blue', 'purple','black','orange'])
	scatter =plt.scatter(x=df_visu['Age'],y=df_visu['Spending Score (1-100)'],c=df_visu['prediction'],cmap=colors)
	plt.legend(handles=scatter.legend_elements()[0], labels=classes)


def pca_visu(df,label):
	u_labels = np.unique(label)
	#plotting the results:
	rcParams['figure.figsize'] = 10, 10
	for i in u_labels:
		plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
	plt.legend()
	plt.show()


def pca_visu_centroid(df,label):	
	#from sklearn.cluster import KMeans
	#kmeans = KMeans(n_clusters= 5)	
	centroids = kmeans.cluster_centers_
	u_labels = np.unique(label)

	#plotting the results:
	for i in u_labels:
		plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
	plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
	plt.legend()
	plt.show()

	#recommandation








	
	

	