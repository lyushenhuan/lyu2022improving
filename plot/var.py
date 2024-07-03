import numpy as np


def ST(mat,label,s):
	"""
	mat : (num_samples,2)
	s : num of class
	"""
	temp_mat = mat.copy()
	mean = np.mean(temp_mat,0)
	temp = temp_mat-mean
	var = np.dot(temp.T, temp)
	st = np.sum(np.diag(var))
	return st

def SE(mat,label,s):
	"""
	mat : (num_samples,2)
	s : num of class
	"""
	temp_mat = mat.copy()
	mean = np.zeros([s,2])
	var = np.zeros(s)
	for i in range(s):
		mean[i] = np.mean(temp_mat[label==i], 0)
		temp_mat[label==i] = temp_mat[label==i] - mean[i]
		var[i] = np.sum(np.diag(np.dot(temp_mat[label==i].T, temp_mat[label==i])))
	se = np.sum(var)
	return se



def SA(mat,label,s):
	"""
	mat : (num_samples,2)
	s : num of class
	"""
	sa = ST(mat,label,s) - SE(mat,label,s)
	return sa
		

if __name__ == '__main__':

	dataset_list = ['CIFAR10','MNIST']
	for dataset in dataset_list:

		test_label = np.loadtxt("./embed/{}_label_test.txt".format(dataset), dtype="int")
		train_label = np.loadtxt("./embed/{}_label_train.txt".format(dataset), dtype="int")
		m1 = train_label.shape[0]
		m2 = test_label.shape[0]
		s = np.max(train_label) + 1

		train_var_se = []
		train_var_sa = []
		train_rate = []
		model_list = ['Xent', 'MLM', 'SMLM', 'ODN']
		for model in model_list:
			mat = np.loadtxt("./embed/{}_{}_train.txt".format(dataset,model))
			temp1 = SE(mat,train_label,s)
			temp2 = SA(mat,train_label,s)
			train_var_se.append(temp1/m1)
			train_var_sa.append(temp2/m1)
			train_rate.append(temp1/temp2)

		test_var_se = []
		test_var_sa = []
		test_rate = []
		for model in model_list:
			mat = np.loadtxt("./embed/{}_{}_test.txt".format(dataset,model))
			temp1 = SE(mat,test_label,s)
			temp2 = SA(mat,test_label,s)
			test_var_se.append(temp1/m2)
			test_var_sa.append(temp2/m2)
			test_rate.append(temp1/temp2)

		print('\n'+dataset)
		print(train_var_se)
		print(train_var_sa)
		print(train_rate)
		print("\n")
		print(test_var_se)
		print(test_var_sa)
		print(test_rate)



