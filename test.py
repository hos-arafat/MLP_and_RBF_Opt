import numpy as np
from models import MLP_Network


def run_me(X, Y):
    NN = MLP_Network(X, Y, X, Y, 13, 1.0, 0.00001, 0)
    NN.train()
    return NN.train_final_loss


dataset = np.genfromtxt("../DATA.csv", delimiter=',')
X = np.array(dataset[1:,:2])
y = np.array(dataset[1:,2])
y = y.reshape( (np.shape(y)[0], 1) )

run_me(X, y)
