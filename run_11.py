#------ Full minimization MLP -------



from models import *


NN = MLP_Network(X_train, y_train, X_test, y_test, 13, 1.0, 0.00001, 0)
NN.train()
NN.surface_plot("Final output of the MLP Network (After Optimization)  \n Full Minimization")
