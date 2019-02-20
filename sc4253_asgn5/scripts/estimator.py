#!/usr/bin/env python

# Assignment 5 
# Siddharth Chamarthy - sc4253

import math
import numpy
import rospy

from geometry_msgs.msg import Pose2D
from state_estimator.msg import SensorData

# Class that contains the extended kalman filter algorithm
class estimator(object):

	def __init__(self):
		self.x = 0
		self.y = 0
		self.th = 0
		self.t = 0.01
		self.P_pred = numpy.zeros([3,3])
		self.X_cap = numpy.zeros([3,1])
		self.P_cap = numpy.zeros([3,3])

		# P covariance noise
		self.V = 0.1*numpy.identity(3)

		# Subscriber for sensor data
		rospy.Subscriber("/sensor_data", SensorData, self.est_callback)

		self.sensor_data = SensorData()

		# publisher for the ExtendedKF
		self.pub_pose = rospy.Publisher("/robot_pose_estimate", Pose2D, queue_size = 1)

		
	# callback for collecting sensor data and for calling the Extended Kalman Filter
	def est_callback(self, data):
		self.sensor_data = data
		self.ExtendedKF(data)

	# main control loop for extended Kalman filter
	def ExtendedKF(self, msg):

		# Collect data from message
		v = msg.vel_trans
		a = msg.vel_ang

		# Create the linearized F matrix
		F = numpy.array([[1,0,-1*self.t*v*numpy.sin(self.th)],
						 [0,1,self.t*v*numpy.cos(self.th)],
						 [0,0,1]])

		F_T = numpy.transpose(F)



		P = numpy.zeros([3,3])
		X = numpy.zeros([3,1])



		# PREDICTION STEPS

		# model of the system dynamics
		x_cap = self.x + self.t*v*numpy.cos(self.th)
		y_cap = self.y + self.t*v*numpy.sin(self.th)
		t_cap = self.th + self.t*a

		self.X_cap[0] = x_cap
		self.X_cap[1] = y_cap
		self.X_cap[2] = t_cap

		# covariance prediction
		self.P_cap = numpy.dot(numpy.dot(F,self.P_pred),F_T) + self.V

		# UPDATE STEPS

		# BUILDING H MATRIX

		# number of sensors in range
		N = len(msg.readings)
		x_l = numpy.empty([N,1])
		y_l = numpy.empty([N,1])

		# storing all the positions of the sensors in range
		for i in range(N):
			x_l[i] = msg.readings[i].landmark.x
			y_l[i] = msg.readings[i].landmark.y	


		# CONDITION: no sensors in range 
		if N == 0:
			X = self.X_cap
			P = self.P_cap


			self.x = X[0]
			self.y = X[1]
			self.th = X[2]
			self.P_pred = P



			# PUBLISHING STEPS
			Obj = Pose2D()
			Obj.x = X[0]
			Obj.y = X[1]
			Obj.theta = X[2]
			self.pub_pose.publish(Obj)


		# CONDITION: 1 sensor in range 
		elif N == 1:

			H = numpy.zeros([2,3])
			y = numpy.zeros([2,1])
			W = 0.1*numpy.identity(2)
			u = numpy.zeros([2,1])

			# Building Sensor Matrix
			H[0][0] = (self.X_cap[0]-x_l[0])/numpy.sqrt(numpy.add(numpy.power((self.X_cap[0]-x_l[0]),2),numpy.power((self.X_cap[1]-y_l[0]),2)))
			H[0][1] = (self.X_cap[1]-y_l[0])/numpy.sqrt(numpy.add(numpy.power((self.X_cap[0]-x_l[0]),2),numpy.power((self.X_cap[1]-y_l[0]),2)))
			H[0][2] = 0
			H[1][0] = (y_l[0]-self.X_cap[1])/(numpy.add(numpy.power((self.X_cap[0]-x_l[0]),2),numpy.power((self.X_cap[1]-y_l[0]),2)))
			H[1][1] = (self.X_cap[0]-x_l[0])/(numpy.add(numpy.power((self.X_cap[0]-x_l[0]),2),numpy.power((self.X_cap[1]-y_l[0]),2)))
			H[1][2] = -1
			
			y[0] = msg.readings[0].range
			y[1] = msg.readings[0].bearing

			H_T = numpy.transpose(H)
			
			S = numpy.add(numpy.dot(numpy.dot(H,self.P_cap),H_T),W)
			S_inverse = numpy.linalg.pinv(S)

			R = numpy.dot(numpy.dot(self.P_cap,H_T),S_inverse)
			
			# Innovation
			u[0] = y[0] - math.sqrt( (self.X_cap[0]-x_l[0])*(self.X_cap[0]-x_l[0]) + (self.X_cap[1]-y_l[0])*(self.X_cap[1]-y_l[0]) )
			u[1] = y[1] - math.atan2(y_l[0]-self.X_cap[1], x_l[0]-self.X_cap[0]) + self.X_cap[2]
	
			X = numpy.add(self.X_cap,numpy.dot(R,u))
			P = numpy.subtract(self.P_cap,numpy.dot(numpy.dot(R,H),self.P_cap))

			self.x = X[0]
			self.y = X[1]
			self.th = X[2]
			self.P_pred = P

			# PUBLISHING STEPS
			Obj = Pose2D()
			Obj.x = X[0]
			Obj.y = X[1]
			Obj.theta = X[2]
			self.pub_pose.publish(Obj)

		# CONDITION: 2 sensors in range
		elif N == 2:

			H = numpy.zeros([4,3])
			y = numpy.zeros([4,1])
			W = 0.1*numpy.identity(4)
			u = numpy.zeros([4,1])

			# Building Sensor Matrix
			H[0][0] = (self.X_cap[0]-x_l[0])/numpy.sqrt(numpy.add(numpy.power((self.X_cap[0]-x_l[0]),2),numpy.power((self.X_cap[1]-y_l[0]),2)))
			H[0][1] = (self.X_cap[1]-y_l[0])/numpy.sqrt(numpy.add(numpy.power((self.X_cap[0]-x_l[0]),2),numpy.power((self.X_cap[1]-y_l[0]),2)))
			H[0][2] = 0
			H[1][0] = (y_l[0]-self.X_cap[1])/(numpy.add(numpy.power((self.X_cap[0]-x_l[0]),2),numpy.power((self.X_cap[1]-y_l[0]),2)))
			H[1][1] = (self.X_cap[0]-x_l[0])/(numpy.add(numpy.power((self.X_cap[0]-x_l[0]),2),numpy.power((self.X_cap[1]-y_l[0]),2)))
			H[1][2] = -1

			H[2][0] = (self.X_cap[0]-x_l[1])/numpy.sqrt(numpy.add(numpy.power((self.X_cap[0]-x_l[1]),2),numpy.power((self.X_cap[1]-y_l[1]),2)))
			H[2][1] = (self.X_cap[1]-y_l[1])/numpy.sqrt(numpy.add(numpy.power((self.X_cap[0]-x_l[1]),2),numpy.power((self.X_cap[1]-y_l[1]),2)))
			H[2][2] = 0
			H[3][0] = (y_l[1]-self.X_cap[1])/(numpy.add(numpy.power((self.X_cap[0]-x_l[1]),2),numpy.power((self.X_cap[1]-y_l[1]),2)))
			H[3][1] = (self.X_cap[0]-x_l[1])/(numpy.add(numpy.power((self.X_cap[0]-x_l[1]),2),numpy.power((self.X_cap[1]-y_l[1]),2)))
			H[3][2] = -1
			
			y[0] = msg.readings[0].range
			y[1] = msg.readings[0].bearing
			y[2] = msg.readings[1].range
			y[3] = msg.readings[1].bearing
			
			H_T = numpy.transpose(H)

			S = numpy.add(numpy.dot(numpy.dot(H,self.P_cap),H_T),W)
			S_inverse = numpy.linalg.inv(S)

			R = numpy.dot(numpy.dot(self.P_cap,H_T),S_inverse)

			# Innovation
			u[0] = y[0] - math.sqrt( (self.X_cap[0]-x_l[0])*(self.X_cap[0]-x_l[0]) + (self.X_cap[1]-y_l[0])*(self.X_cap[1]-y_l[0]) )
			u[1] = y[1] - math.atan2(y_l[0]-self.X_cap[1], x_l[0]-self.X_cap[0]) + self.X_cap[2]
			u[2] = y[2] - math.sqrt( (self.X_cap[0]-x_l[1])*(self.X_cap[0]-x_l[1]) + (self.X_cap[1]-y_l[1])*(self.X_cap[1]-y_l[1]) )
			u[3] = y[3] - math.atan2(y_l[1]-self.X_cap[1], x_l[1]-self.X_cap[0]) + self.X_cap[2]

			X = numpy.add(self.X_cap,numpy.dot(R,u))
			P = numpy.subtract(self.P_cap,numpy.dot(numpy.dot(R,H),self.P_cap))

			self.x = X[0]
			self.y = X[1]
			self.th = X[2]
			self.P_pred = P

			# PUBLISHING STEPS
			Obj = Pose2D()
			Obj.x = X[0]
			Obj.y = X[1]
			Obj.theta = X[2]
			self.pub_pose.publish(Obj)

		# CONDITION: 3 sensors in range
		elif N == 3:

			H = numpy.zeros([6,3])
			y = numpy.zeros([6,1])
			W = 0.1*numpy.identity(6)
			u = numpy.zeros([6,1])

			# Building Sensor Matrix
			H[0][0] = (self.X_cap[0]-x_l[0])/numpy.sqrt(numpy.add(numpy.power((self.X_cap[0]-x_l[0]),2),numpy.power((self.X_cap[1]-y_l[0]),2)))
			H[0][1] = (self.X_cap[1]-y_l[0])/numpy.sqrt(numpy.add(numpy.power((self.X_cap[0]-x_l[0]),2),numpy.power((self.X_cap[1]-y_l[0]),2)))
			H[0][2] = 0
			H[1][0] = (y_l[0]-self.X_cap[1])/(numpy.add(numpy.power((self.X_cap[0]-x_l[0]),2),numpy.power((self.X_cap[1]-y_l[0]),2)))
			H[1][1] = (self.X_cap[0]-x_l[0])/(numpy.add(numpy.power((self.X_cap[0]-x_l[0]),2),numpy.power((self.X_cap[1]-y_l[0]),2)))
			H[1][2] = -1

			H[2][0] = (self.X_cap[0]-x_l[1])/numpy.sqrt(numpy.add(numpy.power((self.X_cap[0]-x_l[1]),2),numpy.power((self.X_cap[1]-y_l[1]),2)))
			H[2][1] = (self.X_cap[1]-y_l[1])/numpy.sqrt(numpy.add(numpy.power((self.X_cap[0]-x_l[1]),2),numpy.power((self.X_cap[1]-y_l[1]),2)))
			H[2][2] = 0
			H[3][0] = (y_l[1]-self.X_cap[1])/(numpy.add(numpy.power((self.X_cap[0]-x_l[1]),2),numpy.power((self.X_cap[1]-y_l[1]),2)))
			H[3][1] = (self.X_cap[0]-x_l[1])/(numpy.add(numpy.power((self.X_cap[0]-x_l[1]),2),numpy.power((self.X_cap[1]-y_l[1]),2)))
			H[3][2] = -1

			H[4][0] = (self.X_cap[0]-x_l[2])/numpy.sqrt(numpy.add(numpy.power((self.X_cap[0]-x_l[2]),2),numpy.power((self.X_cap[1]-y_l[2]),2)))
			H[4][1] = (self.X_cap[1]-y_l[2])/numpy.sqrt(numpy.add(numpy.power((self.X_cap[0]-x_l[2]),2),numpy.power((self.X_cap[1]-y_l[2]),2)))
			H[4][2] = 0
			H[5][0] = (y_l[2]-self.X_cap[1])/(numpy.add(numpy.power((self.X_cap[0]-x_l[2]),2),numpy.power((self.X_cap[1]-y_l[2]),2)))
			H[5][1] = (self.X_cap[0]-x_l[2])/(numpy.add(numpy.power((self.X_cap[0]-x_l[2]),2),numpy.power((self.X_cap[1]-y_l[2]),2)))
			H[5][2] = -1

			y[0] = msg.readings[0].range
			y[1] = msg.readings[0].bearing
			y[2] = msg.readings[1].range
			y[3] = msg.readings[1].bearing
			y[4] = msg.readings[2].range
			y[5] = msg.readings[2].bearing

			H_T = numpy.transpose(H)

			S = numpy.add(numpy.dot(numpy.dot(H,self.P_cap),H_T),W)
			S_inverse = numpy.linalg.inv(S)

			R = numpy.dot(numpy.dot(self.P_cap,H_T),S_inverse)

			# Innovation
			u[0] = y[0] - math.sqrt( (self.X_cap[0]-x_l[0])*(self.X_cap[0]-x_l[0]) + (self.X_cap[1]-y_l[0])*(self.X_cap[1]-y_l[0]) )
			u[1] = y[1] - math.atan2(y_l[0]-self.X_cap[1], x_l[0]-self.X_cap[0]) + self.X_cap[2]
			u[2] = y[2] - math.sqrt( (self.X_cap[0]-x_l[1])*(self.X_cap[0]-x_l[1]) + (self.X_cap[1]-y_l[1])*(self.X_cap[1]-y_l[1]) )
			u[3] = y[3] - math.atan2(y_l[1]-self.X_cap[1], x_l[1]-self.X_cap[0]) + self.X_cap[2]
			u[4] = y[4] - math.sqrt( (self.X_cap[0]-x_l[2])*(self.X_cap[0]-x_l[2]) + (self.X_cap[1]-y_l[2])*(self.X_cap[1]-y_l[2]) )
			u[5] = y[5] - math.atan2(y_l[2]-self.X_cap[1], x_l[2]-self.X_cap[0]) + self.X_cap[2]
		
			X = numpy.add(self.X_cap,numpy.dot(R,u))
			P = numpy.subtract(self.P_cap,numpy.dot(numpy.dot(R,H),self.P_cap))
		
			self.x = X[0]
			self.y = X[1]
			self.th = X[2]
			self.P_pred = P
		
			# PUBLISHING STEPS
			Obj = Pose2D()
			Obj.x = X[0]
			Obj.y = X[1]
			Obj.theta = X[2]
			self.pub_pose.publish(Obj)

if __name__ == '__main__':
	rospy.init_node('estimator', anonymous=True)
	est = estimator()
	rospy.spin()