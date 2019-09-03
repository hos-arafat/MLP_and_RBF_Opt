#------ Two Block Method RBF -------



from models import *


NN = RBF_Network(X_train, y_train, X_test, y_test, 40, 1.0, 0.00001, 1)
NN.extreme_learning()
NN.surface_plot("Final output of the RBF Network (After Optimization) \n 2 Blocks Method")
