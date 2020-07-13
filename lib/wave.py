import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def change_range(data, low, high):
    scaler = MinMaxScaler(feature_range=[low, high])
    scaled = scaler.fit_transform(data.reshape(-1, 1))

    return np.ravel(scaled)

def power(signal):
	n = signal.shape[0]
	sum_sq = np.sum(signal**2)
	return 10*np.log10(sum_sq/n)

class RecursiveLeastSquare:
	def __init__(self, corrupted_sig, desired_sig):
		'''
			_init_ receives corrupted signal and desired signal
			as Numpy array with the same size and both shape
			should be (length,)
		'''
		if corrupted_sig.shape[0] != desired_sig.shape[0]:
			raise Exception(f'Cannot process corrupted signal {corrupted_sig.shape} using desired signal {desired_sig.shape}')
		else:
			self.corr = corrupted_sig
			self.desired = desired_sig
			self.sig_length = desired_sig.shape[0]
			self.anti_noise = np.zeros(self.sig_length)

			'''
				Initialization of default params
				filter_size: size of filter w
				delta = delta in Sd
				L = lambda
			'''
			self.filter_size = 100
			self.delta = 0.1
			self.L = 0.99
			self.result = None
			self.e = np.zeros(self.sig_length)		
		
	def set_params(self, filter_size, delta, L):
		'''
			set_params will (re)set the parameter as the user desire.
		'''
		self.filter_size = filter_size
		self.delta = delta
		self.L = L

	def sq_error(self):
		return self.e**2

	def reduce_noise(self):
		x = np.append(np.zeros(self.filter_size), self.corr)
		s_d = np.eye(self.filter_size)*(1/self.delta)
		w = np.zeros((self.filter_size, 1)) # weight vector

		# RLS Algorithm
		for i in tqdm(range(0, self.sig_length)):
			xk = x[i:i+self.filter_size][::-1]
			xk = xk[np.newaxis].T

			# a priori error
			self.e[i] = self.desired[i] - np.dot(xk.T, w)
			psi = np.dot(s_d, xk) # n vector

			numerator = np.dot(psi, psi.T) # n*n matrix
			denominator = self.L + np.dot(psi.T, xk)

			gain = numerator/denominator
			
			# update autocorrelation matrix
			s_d = (1/self.L)*(s_d - gain)
			dw = np.dot(self.e[i]*s_d, xk)
			w = w + dw # update weight

			# output (a posteriori error)
			self.anti_noise[i] = np.dot(w.T, xk)

		self.result = self.desired - self.anti_noise
		return self.result