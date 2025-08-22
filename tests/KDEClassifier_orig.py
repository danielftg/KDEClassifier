import math
import numpy as np

class kde:
	def __init__(self, mybandwidth=1.0):
		'''
		Initalize the parameter of bandwidth, sample data and sample labels.
		'''
		self.bandwidth = mybandwidth
		self.samples = []
		self.labels = []

	def fit(self, trainsamples, trainlabels):
		'''
		Fit the model using 'trainsamples' as training data and 'trainlabels' as target values
		Parameters:
			trainsamples: Training data, if array or matrix, the shape should be [n_samples, n_features]. 
			trainlabels: Target values,  the shape should be [n_samples].
		'''
		self.trainsamples = trainsamples
		self.trainlabels = trainlabels
		self.trainlabels_set = set(self.trainlabels)
		self.trainlabels_set = list(self.trainlabels_set)
		self.trainlabels_set.sort()
		return self
	
	def set_bandwidth_default(self, trainsamples):
		'''
		set the default bandwidth.
		Parameters:
			trainsamples: Training data, if array or matrix, the shape should be [n_samples, n_features]. 
		'''
		std_list = trainsamples.std(axis=0)
		bandwidth_list = 1.059 * std_list * math.pow(trainsamples.shape[0],-0.2)
		bandwidth_default = np.mean(bandwidth_list)
		self.bandwidth = bandwidth_default

	def __dist(self,X,Y):
		'''
		private function: calculate the Euclidean distance between two points.
		'''
		dist = 0.0
		for x,y in zip(X,Y):
			dist += (x-y)**2
		dist = math.sqrt(dist)
		return dist

	def predict_proba(self, testsamples):
		'''
		Return probability estimates for the test data 'testsamples'
		Parameters:
			testsamples: test data, array-like, shape [n_query, n_features]
		Return:
			predict_probs: The class probabilities of the input samples. Classes are ordered by lexicographic order.
			array-like, shape [n_query, n_classes]
		'''
		predict_probs = []
		for expect in testsamples:
			probability_expects = dict()
			prob_sum = 0.0
			for label in set(self.trainlabels):
				X = []
				for i in range(len(self.trainsamples)):
					if self.trainlabels[i] == label:
						X.append(self.trainsamples[i])
				
				probability_expect = 0
				for x in X:
					probability_x = np.exp(-self.__dist(expect, x)**2/(2*self.bandwidth**2))/(math.sqrt(2*math.pi)*self.bandwidth)
					probability_expect += probability_x
				probability_expect = probability_expect/len(X)
				probability_expects[label] = probability_expect
				prob_sum += probability_expect

			relative_prob = dict()
			for label in probability_expects.keys():
				relative_prob[label] = probability_expects[label]/prob_sum
			
			relative_prob = sorted(relative_prob.items(), key=lambda x:x[0])
			relative_prob = dict(relative_prob)
			relative_prob_value = relative_prob.values()
			relative_prob_value = list(relative_prob_value)
			predict_probs.append(relative_prob_value)

		predict_probs = np.array(predict_probs)
		return predict_probs

	def predict(self, testsamples):
		'''
		Predict the class labels for the provided data 'testsamples'.
		Parameters:
			testsamples: Test samples, array-like, shape [n_query, n_features].
		Return:
			predict_labels: Class labels for each data sample, array-like, shape [n_query].
		'''
		predict_labels = []
		predict_probs = self.predict_proba(testsamples)
		for elem in predict_probs:
			maxindex  = np.argmax(elem)
			predict_labels.append(self.trainlabels_set[maxindex])
		
		predict_labels = np.array(predict_labels)
		return predict_labels