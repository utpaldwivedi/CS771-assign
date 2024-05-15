import numpy as np
from sklearn.linear_model import LogisticRegression

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# X_train has 32 columns containing the challeenge bits
	# y_train contains the responses
	
	# THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
	# If you do not wish to use a bias term, set it to 0
 
	features = my_map(X_train)
	clf = LogisticRegression( solver ="lbfgs", C=1, tol = 0.005)
	clf.fit(features,y_train)
	
	w = clf.coef_[0]
	b = clf.intercept_[0]
	return w, b


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
	D = np.flip(1-2*X, axis = 1)	## re-encode {0,1} challenges to {1,-1} challenges and flip the order
	X1 = np.cumprod(D, axis = 1)	## first order features
	X1 = np.flip(X1,axis = 1)       ## flip just for personal satisfaction :)
	l = len(X1[0])					
	X2 = np.zeros( (len(X1), int(l*(l-1)/2) ) ) 	##stores second order terms
	for i in range(len(X2)):
		so = np.outer(X1[i],X1[i])		##find outer product of 1d features of this challenge
		so = np.triu(so,k=1)			## make zero all elements on and below principal diagonal
		so = so.flatten()				## make 1d vector from upper triangular matrix
		so = so[so!=0]					## keep only non zero part
		X2[i] = so
	feat = np.concatenate((X1,X2),axis = 1)	## take both 1st order and second order features
	
	return feat
