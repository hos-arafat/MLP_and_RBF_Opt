#------ Decomposition method -------



from models import *
#from OMML_ASS_1 import *


#Decide Network Type
#Net_Type = "MLP"
Net_Type = "RBF"

if (Net_Type == "MLP"):
	NN = MLP_Network(X_train, y_train, X_test, y_test, 37, 1.0, 0.00001, 2)
	#NN = Neural_Network(X_train, y_train, 30, 1.0, 0.00001, 2)

elif (Net_Type == "RBF"):
	NN = RBF_Network(X_train, y_train, X_test, y_test, 40, 1.0, 0.00001, 2)


NN.block_Decomposition()
NN.surface_plot("Final output of " + Net_Type + " Network (After Optimization) \n Block Decomposition Method")
