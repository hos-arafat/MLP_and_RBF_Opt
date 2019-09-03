from models import *
"""
loss_values = {}
for N in range(1, 3):
    print("\n")
    print("For this number Neurons", N)
    for sigma in np.arange(0.1, 1.0, 0.7):
        print("--This value of Sigma", sigma)
        for reg in np.arange(0.00001, 0.001, 0.0009):
            print("---This Regularization:", reg)
            NN = MLP_Network(X_train, y_train, X_test, y_test, N, sigma, 0.00001, 0)
            NN.train()
            print("\n")
            loss_values.update({"[N][sigma][reg]": NN.train_final_loss})
            #loss_values[N][sigma][reg] = NN.train_final_loss
            #print("Final Loss:", NN.train_final_loss)
            #loss_values.append(curr_loss)
"""
loss_values = []
for N in range(1, 100):
    NN = MLP_Network(X_train, y_train, X_test, y_test, N, 1.0, 0.00001, 0)
    NN.train()
    curr_loss = NN.train_final_loss
    loss_values.append(curr_loss)

best = np.argmin(loss_values) + 1
print("Best value is", best)
print(loss_values)
