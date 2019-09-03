#------ Full minimization RBF -------



from models import *


NN = RBF_Network(X_train, y_train, X_test, y_test, 30, 1.0, 0.00001, 0)
NN.train()
NN.surface_plot("Final output of the RBF Network (After Optimization)")
