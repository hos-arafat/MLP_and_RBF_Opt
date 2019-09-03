import csv
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
import scipy.optimize as opt
import time
from numpy import linalg as LA

np.random.seed(1803850)

dataset = np.genfromtxt("../DATA.csv", delimiter=',')
X = np.array(dataset[1:,:2])
y =np.array(dataset[1:,2])
y = y.reshape( (np.shape(y)[0], 1) )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1803850)





#======================================MLP==============================================


class MLP_Network(object):
	def __init__(self, X, Y, X_t, Y_t, N=3, sigma=1.0, rho=0.5, mode=0):
		#parameters
		self.inputSize = 2
		self.outputSize = 1
		self.hiddenSize = N

		self.sigma = sigma
		self.rho = rho
		print("Number of neurons N:", self.hiddenSize)

		self.X = X
		self.Y = Y
		self.temp = self.X
		self.X_t = X_t
		self.Y_t = Y_t

		self.mode = mode
		self.solver = "BFGS"

		#weights
		self.W = np.random.randn(self.inputSize, self.hiddenSize) # Weight matrix from input to hidden layer
		self.B = np.random.randn(1, self.hiddenSize) # Bias vector for hidden layer

		self.V = np.random.randn(self.hiddenSize, self.outputSize) # Weight matrix from hidden to output layer

	def forward(self, omega_vec):

		if (self.mode==0):
			V_vector = omega_vec[:self.hiddenSize]
			W_vector = omega_vec[self.hiddenSize : 3*self.hiddenSize]
			B_vector = omega_vec[3*self.hiddenSize:]

			self.V =  V_vector.reshape(self.hiddenSize, 1) # Shape is N x 1
			self.W =  np.reshape(W_vector, (self.inputSize, self.hiddenSize)) # Shape is 2 x N
			self.B  = B_vector.reshape(1, self.hiddenSize)

		elif (self.mode==1):

			self.V =  omega_vec.reshape(self.hiddenSize, 1) # Shape is N x 1

		else:
			W_vector = omega_vec[:2*self.hiddenSize]
			B_vector = omega_vec[2*self.hiddenSize:]

			self.W =  np.reshape(W_vector, (self.inputSize, self.hiddenSize)) # Shape is 2 x N
			self.B  = B_vector.reshape(1, self.hiddenSize)



		#forward propagation through our network
		self.z1 = np.dot(self.X, self.W) - self.B
		self.a1 = self.tanh(self.z1) # activation function
		self.z2 = np.dot(self.a1, self.V)



	def train(self):
		#backward propagation through our network
		V_vector  = self.V
		W_vector  = np.reshape(self.W, (2*self.hiddenSize,1))
		B_vector  = np.transpose(self.B)

		omega_vec = np.concatenate([V_vector, W_vector, B_vector], 0) # omega = (V, B, W)

		print("Initial training error:", self.reg_loss(omega_vec))
		time_start = time.time()
		result = opt.minimize(self.reg_loss, omega_vec, method = "BFGS")
		time_end = time.time()
		optimal_omega = result.x
		print("Final training error:", self.loss(optimal_omega))
		self.train_final_loss = self.loss(optimal_omega)
		print("Final test error:", self.test_loss(optimal_omega))
		print("Optimization Solver chosen:", self.solver)
		print("Norm of the gradient at the optimal point:", np.linalg.norm(result.jac))
		print("Time for optimizing the network:", time_end - time_start)
		print("Number of function evaluations:", result.nfev)
		print("Value of sigma:", self.sigma)
		print("Value of rho:", self.rho)
		print("Termination reason",result.message)


	def test_loss(self, omega_vec):
		self.X = self.X_t
		self.forward(omega_vec)
		return np.mean(np.square(self.z2-self.Y_t))

	def tanh(self, z):
		U= 1 - np.exp((-self.sigma *2*z))
		D= 1 + np.exp((-self.sigma *2*z))

		return U/D


	def loss(self, omega_vec):

		self.forward(omega_vec)
		#print(np.mean(np.square(self.z2-self.Y)))
		return np.mean(np.square(self.z2-self.Y))


	def reg_loss(self, omega_vec):

		self.forward(omega_vec)
		#print(np.mean(np.square(self.z2-self.Y)) + self.rho*(np.square(LA.norm(omega_vec))))
		return np.mean(np.square(self.z2-self.Y)) + self.rho*(np.square(LA.norm(omega_vec)))


	def extreme_learning(self, disp):

		V_vector  = self.V

		print("Initial training error:", self.loss(V_vector))
		time_start = time.time()
		result = opt.minimize(self.reg_loss, V_vector, method = "BFGS")
		time_end = time.time()
		self.V = result.x
		self.train_final_loss = self.loss(self.V)
		if(disp):
			print("Final Training Error:", self.loss(self.V))
			print("Final test error:", self.test_loss(self.V))
			print("Optimization Solver chosen:", self.solver)
			print("Norm of the gradient at the optimal point:", np.linalg.norm(result.jac))
			print("Time for optimizing the network:", time_end - time_start)
			print("Number of function evaluations:", result.nfev)
			print("Termination reason",result.message)
			print("Value of sigma:", self.sigma)
			print("Value of rho:", self.rho)




	def block_Decomposition(self):

		W_vector  = np.reshape(self.W, (2*self.hiddenSize,1))
		B_vector  = np.transpose(self.B)
		self.mode = 0
		omega_vec = np.concatenate([self.V, W_vector, B_vector], 0) # omega = (V, B, W)

		stop = True
		prev_loss = self.loss(omega_vec)
		#print("Initial training error:", prev_loss)

		# while(stop):
		for i in range(1):
			time_start = time.time()

			self.mode=1
			self.extreme_learning(False)


			#print("step 2 start")
			self.mode=2
			W_vector  = np.reshape(self.W, (2*self.hiddenSize,1))
			B_vector  = np.transpose(self.B)
			optimal_omega_vec = np.concatenate([W_vector, B_vector], 0) # omega = (V, B, W)

			result2 = opt.minimize(self.reg_loss, optimal_omega_vec, method = "L-BFGS-B")

			optimal_omega_vec = result2.x
			W_vector = optimal_omega_vec[:2*self.hiddenSize]
			B_vector = optimal_omega_vec[2*self.hiddenSize:]

			# self.V =  self.V.reshape(self.hiddenSize, 1) # Shape is N x 1
			self.W =  np.reshape(W_vector, (self.inputSize, self.hiddenSize)) # Shape is 2 x N
			self.B  = B_vector.reshape(1, self.hiddenSize)

			#print("step 2 done")

		time_end = time.time()

		optimal_omega = result2.x
		print("Final Training Error:", self.loss(optimal_omega))
		self.train_final_loss = self.loss(optimal_omega)
		print("Final Test Error:", self.test_loss(optimal_omega))
		self.solver = "L-BFGS-B"
		print("Optimization Solver chosen:", self.solver)
		print("Norm of the gradient at the optimal point:", np.linalg.norm(result2.jac))
		print("Time for optimizing the network:", time_end - time_start)
		print("Number of function evaluations:", result2.nfev)
		print("Termination reason",result2.message)
		print("Value of sigma:", self.sigma)
		print("Value of rho:", self.rho)


	def surface_plot(self, title):

		fig = plt.figure()
		fig.patch.set_facecolor('white')
		ax = fig.add_subplot(111, projection = '3d')

		V_vector  = np.reshape(self.V, (np.shape(self.V)[0],1))
		W_vector  = np.reshape(self.W, (2*self.hiddenSize,1))
		B_vector  = np.transpose(self.B)

		optimal_parameters = np.concatenate([V_vector, W_vector, B_vector], 0) # omega = (V,C)

		self.mode = 0
		self.X = self.temp
		self.forward(optimal_parameters)

		ax.scatter(self.X[:,0],self.X[:,1],np.ravel(self.z2), color="red", alpha = 1)
		# ax.plot_trisurf(self.X[:,0],self.X[:,1],np.ravel(self.z2), color="red", alpha = 0.8)
		ax.plot_trisurf(self.X[:,0],self.X[:,1],np.ravel(self.z2), cmap=plt.cm.get_cmap("gist_heat"), alpha = 0.8)
		# ax.plot_trisurf(self.X[:,0],self.X[:,1],np.ravel(self.Y), color = "green", alpha = 0.5)
		ax.set_title(title)
		plt.show()
		plt.close()











































#======================================RBF==============================================

class RBF_Network(object):
	def __init__(self, X, Y, X_t, Y_t, N=3, sigma=np.sqrt(2), rho=0.5, mode = 0):
		#parameters
		self.inputSize = 2
		self.outputSize = 1
		self.hiddenSize = N
		print("Number of neurons N:", self.hiddenSize)

		self.solver = "BFGS"

		self.sigma = sigma
		self.rho = rho

		self.mode = mode

		self.X = X
		self.Y = Y
		self.temp = self.X
		self.X_t = X_t
		self.Y_t = Y_t
		self.disp = True

		self.P = self.X.shape[0]
		# JUST CHANGE THE SHAPE OF P to shape of the current X_test !
		self.P_t = self.X_t.shape[0]
		self.training = True

		#weights
		self.C1 = np.random.randn(self.inputSize, self.hiddenSize) # Weight matrix from input to hidden layer
		self.V = np.random.randn(self.hiddenSize, self.outputSize) # Weight matrix from hidden to output layer



	def forward(self, omega_vec):

		if (self.mode==0):
			V_vector = omega_vec[:self.hiddenSize]
			C1_vector = omega_vec[self.hiddenSize:]

			self.V =  V_vector.reshape(self.hiddenSize, self.outputSize) # Shape is N x 1
			self.C1 =  np.reshape(C1_vector, (self.inputSize, self.hiddenSize)) # Shape is 2 x N
			if(self.training):
				self.C = np.tile(self.C1, (self.P, 1, 1))
			else:
				self.C = np.tile(self.C1, (self.P_t, 1, 1))

		elif (self.mode==1):
			self.V =  omega_vec.reshape(self.hiddenSize, 1) # Shape is N x 1
			if(self.training):
				self.C = np.tile(self.C1, (self.P, 1, 1))
			else:
				self.C = np.tile(self.C1, (self.P_t, 1, 1))

		else:
			self.C1 =  np.reshape(omega_vec, (self.inputSize, self.hiddenSize)) # Shape is 2 x N
			if(self.training):
				self.C = np.tile(self.C1, (self.P, 1, 1))
			else:
				self.C = np.tile(self.C1, (self.P_t, 1, 1))


		if(self.training):
			z_output = np.zeros((self.P, self.hiddenSize))
		else:
			z_output = np.zeros((self.P_t, self.hiddenSize))

		for n in range(self.hiddenSize):
			gauss_input = self.X - self.C[:, :, n]
			z_output[:,n] = self.Gaus(np.linalg.norm(gauss_input, 2, axis=1))

		self.z2 = np.dot(z_output, self.V)


	def train(self):
		V_vector  = self.V
		C_vector  = np.reshape(self.C1, (2*self.hiddenSize,1))

		omega_vec = np.concatenate([V_vector, C_vector], 0) # omega = (V,C)

		print("Initial Training Error:", self.reg_loss(omega_vec))
		time_start = time.time()
		result = opt.minimize(self.reg_loss, omega_vec, method = "BFGS")
		time_end = time.time()
		optimal_omega = result.x
		self.train_final_loss = self.loss(optimal_omega)
		print("Final Training Error:", self.loss(optimal_omega))
		print("Final Test Error:", self.test_loss(optimal_omega))
		print("Optimization Solver chosen:", self.solver)
		print("Norm of the gradient at the optimal point:", np.linalg.norm(result.jac))
		print("Time for optimizing the network:", time_end - time_start)
		print("Number of function evaluations:", result.nfev)
		print("Value of sigma:", self.sigma)
		print("Value of rho:", self.rho)
		print("Termination reason",result.message)

	def test_loss(self, omega_vec):
		self.X = self.X_t
		#self.P = self.X.shape[0]
		self.training = False
		self.forward(omega_vec)
		return np.mean(np.square(self.z2-self.Y_t))

	def Gaus(self, z):

		a= np.exp(-(np.square(z/self.sigma)))
		return a

	def loss(self, omega_vec):

		self.forward(omega_vec)
		#print(np.mean(np.square(self.z2 - self.Y)))
		return np.mean(np.square(self.z2 - self.Y))

	def reg_loss(self, omega_vec):

		self.forward(omega_vec)
		return np.mean(np.square(self.z2-self.Y)) + self.rho*(np.square(LA.norm(omega_vec)))


	def extreme_learning(self):

		V_vector  = self.V
		if(self.disp):
			print("Initial Training Error", self.loss(V_vector))
		time_start = time.time()
		result = opt.minimize(self.reg_loss, V_vector, method = "BFGS")
		time_end = time.time()


		optimal_omega = result.x
		self.V = optimal_omega
		self.train_final_loss = self.loss(optimal_omega)
		if(self.disp):
			print("Final Trainig Error", self.loss(optimal_omega))
			print("Final Test Error", self.test_loss(optimal_omega))
			print("Optimization solver chosen:", self.solver)
			print("Norm of the gradient at the optimal point:", np.linalg.norm(result.jac))
			print("Time for optimizing the network:", time_end - time_start)
			print("Number of function evaluations:", result.nfev)
			print("Value of sigma:", self.sigma)
			print("Value of rho:", self.rho)
			print("Termination reason",result.message)




	def block_Decomposition(self):

		V_vector  = self.V
		C_vector  = np.reshape(self.C1, (2*self.hiddenSize,1))

		omega_vec = np.concatenate([V_vector, C_vector], 0) # omega = (V,C)
		self.mode = 0
		print("Initial Training Error", self.reg_loss(omega_vec))
		stop = True
		prev_loss = self.loss(omega_vec)

		for i in range(1):

			time_start = time.time()

			self.mode=1
			self.disp = False
			self.extreme_learning()


			self.mode=2
			optimal_omega_vec = np.reshape(self.C1, (2*self.hiddenSize,1))
			result2 = opt.minimize(self.reg_loss, optimal_omega_vec, method = "L-BFGS-B")
			optimal_omega_vec = result2.x
			C_vector = optimal_omega_vec
			self.C1  = C_vector.reshape(self.inputSize, self.hiddenSize)


		time_end = time.time()
		optimal_omega = result2.x
		print("Final Training Error:", self.loss(optimal_omega))
		self.train_final_loss = self.reg_loss(optimal_omega)
		print("Final Test Error:", self.test_loss(optimal_omega))
		self.solver = "L-BFGS-B"
		print("Optimization Solver chosen:", self.solver)
		print("Norm of the gradient at the optimal point:", np.linalg.norm(result2.jac))
		print("Time for optimizing the network:", time_end - time_start)
		print("Number of function evaluations:", result2.nfev)
		# print("Termination reason",result2.message)
		print("Value of sigma:", self.sigma)
		print("Value of rho:", self.rho)

	def surface_plot(self, title):

		fig = plt.figure()
		fig.patch.set_facecolor('white')
		ax = fig.add_subplot(111, projection = '3d')

		V_vector  = np.reshape(self.V, (np.shape(self.V)[0],1))
		C_vector  = np.reshape(self.C1, (2*self.hiddenSize,1))

		optimal_parameters = np.concatenate([V_vector, C_vector], 0) # omega = (V,C)

		self.X = self.temp
		self.training = True
		self.mode = 0

		self.forward(optimal_parameters)

		ax.scatter(self.X[:,0],self.X[:,1],np.ravel(self.z2), color="red", alpha = 1)
		#ax.plot_trisurf(self.X[:,0],self.X[:,1],np.ravel(self.z2), color="red", alpha = 0.8)
		ax.plot_trisurf(self.X[:,0],self.X[:,1],np.ravel(self.z2), cmap=plt.cm.get_cmap("gist_heat"), alpha = 0.8)
		# ax.plot_trisurf(self.X[:,0],self.X[:,1],np.ravel(self.Y), color = "green", alpha = 0.5)
		ax.set_title(title)
		plt.show()
		plt.close()




























# # NN_1 = Neural_Network(N=3, sigma=1.0, alpha=0.001, rho=0.5)
# NN_2 = MLP_Network(X_train, y_train, 60, 1.0, 0.00001, 1)
# NN_1 = RBF_Network(X_train, y_train, 150, 1.0, 0.00001, 2)
# #NN_2.extreme_learning()
# NN_1.block_Decomposition()
# #NN_1.train()
# NN_1.surface_plot("Final output of the Network (After Optimization)")

# # for N in range(30,31):
# # 	NN_1 = MLP_Network(X_train, y_train, N, 1.0, 0.00001, 0)
# # 	NN_1.train()
# # 	print("Number of neurons:", N)
# # 	print("Final loss:", NN_1.train_final_loss)
