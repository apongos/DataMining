import itertools
import timeit
import collections
import math
import random
import numpy
from scipy.linalg import norm

Max=float("-inf")
Min=float("inf")
NormDict = {}

def cluster(InputFile, k, OutPutName):
	#Create a list of items to then normalize the data
	f = open(InputFile, 'r')
	ID = 0
	global Max
	global Min
	FileDict = {}
	for line in f:
		ID += 1
		fileLineTuple = []
		items = line.split()
		for item in items:
			item2 = float(item)
			if item2 < Min:
				Min = item2
			if item2 > Max:
				Max = item2	
			fileLineTuple.append(item)
		fileLineTuple = tuple(fileLineTuple)
		FileDict[ID] = fileLineTuple

	# Normalize the data
	global NormDict
	NormDict = MinMaxNorm(FileDict)

	#Get random centroids
	centroidDict = {}
	tups = NormDict.values()
	attlen = len(NormDict[1]) #cluster dimension has to match input dimension
	for x in range(0,k):
		centroid = random.choice(tups)
		centroidDict[x] = tuple(centroid)

	print"Starting Centroids", centroidDict

	#Assign tuples to clusters
	ClustersObjects = assign_to_cluster(NormDict, centroidDict)
	print"Iteration1 Centroid Initial Objects", ClustersObjects	

	#For all objects in a cluster, find new centroids by determining the mean of those objects
	oldCentroids = centroidDict
	newCentroids = findCentroidMean(ClustersObjects)

	# Iterate until no new assignments
	newAssignments = ''
	iteration = 0
	while has_converged(oldCentroids,newCentroids)==False:
		iteration +=1 
		oldCentroids = newCentroids
		newAssignments = assign_to_cluster(NormDict, newCentroids)
		newCentroids = findCentroidMean(newAssignments)
		print"Iteration %d Centroid New Objects" %iteration, newAssignments
	print"Final Centroids", newCentroids

	#Make obj belongs to centroid; ordered list of tuples (objID, centroidID)
	belongsTo = []
	for key, lists in newAssignments.iteritems():
		for item in lists:		
			belongsTo.append( (item, key) )
	belongsTo.sort(key = lambda x: x[0])

	print belongsTo
	belongsTo2 = []
	belongsTo3 = []
	for tuples in belongsTo:
		belongsTo2.append(tuples[1])
		belongsTo3.append(tuples[0])
	print"BT2",belongsTo2
	print"BT3",belongsTo3
	#Attempt visualization
	import os  # for os.path.basename
	import matplotlib.pyplot as plt
	import matplotlib as mpl
	from sklearn.manifold import MDS
	MDS()

	# convert two components as we're plotting points in a two-dimensional plane
	# "precomputed" because we provide a distance matrix
	# we will also specify `random_state` so the plot is reproducible.
	dist = NormDict.values()
	from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
	dist = 1 - cosine_similarity(dist)
	#dist = 1 - euclidean_distances(dist)
	mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
	pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
	xs, ys = pos[:, 0], pos[:, 1]
	print"sizex, sizey", len(xs), len(ys)
	
	#set up colors per clusters using a dict
	cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3'}
	
	#set up cluster names using a dict
	cluster_names = {0: 'Cluster0', 
	                 1: 'Cluster1', 
	                 2: 'Cluster2', }

	#create data frame that has the result of the MDS plus the cluster numbers and titles)
	import pandas as pd
	df = pd.DataFrame(dict(x=xs, y=ys, label=belongsTo2, title=belongsTo3)) 

	#group by cluster
	groups = df.groupby('label')


	# set up plot
	fig, ax = plt.subplots(figsize=(17, 9)) # set size
	ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

	#iterate through groups to layer the plot
	#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
	for name, group in groups:
	    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
	            label=cluster_names[name], color=cluster_colors[name], 
	            mec='none')
	    ax.set_aspect('auto')
	    ax.tick_params(\
	        axis= 'x',          # changes apply to the x-axis
	        which='both',      # both major and minor ticks are affected
	        bottom='off',      # ticks along the bottom edge are off
	        top='off',         # ticks along the top edge are off
	        labelbottom='off')
	    ax.tick_params(\
	        axis= 'y',         # changes apply to the y-axis
	        which='both',      # both major and minor ticks are affected
	        left='off',      # ticks along the bottom edge are off
	        top='off',         # ticks along the top edge are off
	        labelleft='off')
	    
	ax.legend(numpoints=1)  #show legend with only 1 point

	#add label in x,y position with the label as the film title
	for i in range(len(df)):
	    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  

	    
	    
	plt.show() #show the plot

	#uncomment the below to save the plot if need be
	#plt.savefig('clusters_small_noaxes.png', dpi=200)

def assign_to_cluster(NormDict2, centroidDict):
	global NormDict
	NormDict = NormDict2
	closest_cluster = {}
	ClustersObjects = {} #clusterID: SetOfObjects
	#For every object, find distances to centroids
	for key, obj in NormDict.iteritems():
		compareCentroid = {}
		for clusterID, centroid in centroidDict.iteritems():
			subt = numpy.subtract(centroid, obj)
			subt = norm(subt)
			if clusterID not in compareCentroid.keys():
				compareCentroid[clusterID] = []
				compareCentroid[clusterID].append(subt)
			else:
				compareCentroid[clusterID].append(subt)
			#print"compareCentroid", compareCentroid
		#Now compare centroid and choose the closest for the obj
		minkey = min(compareCentroid, key = compareCentroid.get)
		#print"minkey", minkey
		if minkey not in ClustersObjects.keys():
			ClustersObjects[minkey] = []
			ClustersObjects[minkey].append(key) 
		else:
			ClustersObjects[minkey].append(key) 
	return ClustersObjects


def has_converged(old, new):
	print "has converged?", set(old.values()) == set(new.values())
	return set(old.values()) == set(new.values())

def findCentroidMean(ClustersObjects):
	global NormDict
	ClusterMeans = {}
	# print "ClustersObjects.keys()", ClustersObjects.keys()
	# print"ClustersObjects.values", ClustersObjects.values()
	# print"NormDict[ID]",NormDict[1]
	tupleOfTups = tuple()
	for cluster, List in ClustersObjects.iteritems():
		for ID in List:
			tupleOfTups += (NormDict[ID],) 
		ClusterMeans[cluster] = tuple(numpy.mean(tupleOfTups, axis = 0)) 
	# print"ClusterMeans", ClusterMeans
	return ClusterMeans


def MinMaxNorm(FileDict):
	global Max
	global Min
	NormDict = {}
	for key, tups in FileDict.iteritems():
		fileLineTuple = []
		for item in tups:
			item2 = float(item)
			value = (item2 - Min)/(Max-Min)
			fileLineTuple.append(value)
		fileLineTuple = tuple(fileLineTuple)
		NormDict[key] = fileLineTuple
	#print"In Function", NormDict
	return NormDict

#Main Method 
if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Cluster')
	parser.add_argument('InputFile', help = "type in seeds.txt as file name", type=str)
	parser.add_argument('K', help='type integer', type=int)
	parser.add_argument('OutFile', help = "type name for output. ie Result.txt as file name", type=str)


	args = parser.parse_args()

	start_time = timeit.default_timer()
	cluster(args.InputFile, args.K, args.OutFile)
	elapsed = timeit.default_timer() - start_time
	print"elapsed time", elapsed







