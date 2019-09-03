#------ Two Block Method MLP -------



from models import *


NN = MLP_Network(X_train, y_train, X_test, y_test, 37, 1.0, 0.00001, 1)
NN.extreme_learning(True)
NN.surface_plot("Final output of the MLP Network (After Optimization)  \n 2 Blocks Method")
