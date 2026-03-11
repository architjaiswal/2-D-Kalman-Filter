
'''
Kalman filter without the control input term (B*u) in the state prediction step, allowing the bias state to absorb unknown accelerations. This is a common approach when the control inputs are not known or are difficult to model accurately.
'''

import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter_upgraded(object):
    def __init__(self, dt, std_acc, x_std_meas, y_std_meas):
        """
        :param dt: sampling time (time for 1 cycle)
        :param std_acc: process noise magnitude
        :param x_std_meas: standard deviation of the measurement in x-direction
        :param y_std_meas: standard deviation of the measurement in y-direction
        """

        # Define sampling time
        self.dt = dt

        # Define the  control input variables
        self.u = np.matrix([[0],[0]])  # No control input; bias absorbs it ------

        # Intial State
        self.x = np.matrix([[0], [0], [0], [0], [0], [0]])  # Extended state: [position, velocity, bias_acceleration]

        # Define the State Transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0, self.dt**2/2, 0],
                            [0, 1, 0, self.dt, 0, self.dt**2/2],
                            [0, 0, 1, 0, self.dt, 0],
                            [0, 0, 0, 1, 0, self.dt],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])

        # NO NEED for control matrix B since bias state absorbs unknown accelerations

        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0 ,0, 0],
                            [0, 1, 0, 0, 0, 0]])

        #Initial Process Noise Covariance
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0, (self.dt**2)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2, 0, (self.dt**2)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0, self.dt, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2, 0, self.dt],
                            [(self.dt**2)/2, 0, self.dt, 0, 1, 0],
                            [0, (self.dt**2)/2, 0, self.dt, 0, 1]]) * std_acc**2

        #Initial Measurement Noise Covariance
        self.R = np.matrix([[x_std_meas**2,0],
                           [0, y_std_meas**2]])

        #Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])

    def predict(self):

        # Update time state
        #x_k =Ax_(k-1) 
        self.x = np.dot(self.A, self.x) 

        # Calculate error covariance
        # P= A*P*A' + Q               Eq.(10)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0:2]

    def update(self, z):

        # Refer to :Eq.(11), Eq.(12) and Eq.(13)  in https://machinelearningspace.com/object-tracking-simple-implementation-of-kalman-filter-in-python/?preview_id=1364&preview_nonce=52f6f1262e&preview=true&_thumbnail_id=1795
        # S = H*P*H'+R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  #Eq.(11)

        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))   #Eq.(12)

        I = np.eye(self.H.shape[1])

        # Update error covariance matrix
        self.P = (I - (K * self.H)) * self.P   #Eq.(13)
        return self.x[0:2]
