#!/usr/bin/env python

# Assignment 5 - Particle Filter 
# Siddharth Chamarthy - sc4253

import math
import numpy
import rospy

from geometry_msgs.msg import Pose2D
from state_estimator.msg import SensorData

class state(object):
	X_s = numpy.empty([3,1])
	weight = []



class ParticleFilter(object):
	
	def __init__(self):
		self.Par = 10
		self.x = numpy.zeros([self.Par,1])
		self.y = numpy.zeros([self.Par,1])
		self.th = numpy.zeros([self.Par,1])
		self.t = 0.01
		self.P_pred = numpy.zeros([3,3])
		# self.X_cap = numpy.zeros([3,self.Par])
		self.P_cap = numpy.zeros([3,3])


		# Subscriber for sensor data
		rospy.Subscriber("/sensor_data", SensorData, self.est_callback)

		self.sensor_data = SensorData()

		# publisher for the Particle FIlter
		self.pub_pose = rospy.Publisher("/robot_pose_estimate", Pose2D, queue_size = 1)

		
	# callback for collecting sensor data and for calling the Particle Filter
	def est_callback(self, data):
		self.sensor_data = data
		self.ParticleFilter(data)

	def ParticleFilter(self, msg):

		v = msg.vel_trans
		a = msg.vel_ang
		X = numpy.zeros([3,1])
		X_cap = list()
		X_new = list()
		for i in range(self.Par):
			obj = state()
			obj_new = state()
			X_cap.append(state())
			X_new.append(state())

		# PREDICTION STEPS
		
		for i in range(self.Par):
			# model of the system dynamics
			X_cap_data = numpy.empty([3,1])
			x_noise = numpy.random.uniform(0,1)
			y_noise = numpy.random.uniform(0,1)
			t_noise = numpy.random.uniform(0,1)
			x_cap = self.x[i] + self.t*v*numpy.cos(self.th[i]) + x_noise
			y_cap = self.y[i] + self.t*v*numpy.sin(self.th[i]) + y_noise
			t_cap = self.th[i] + self.t*a + t_noise
			X_cap_data[0] = x_cap
			X_cap_data[1] = y_cap
			X_cap_data[2] = t_cap
			X_cap[i].X_s = X_cap_data
			X_cap[i].weight = 1.0/self.Par

		# number of sensors in range
		N = len(msg.readings)

		if N == 0:
			X_mean = numpy.empty([3,1])
			for i in range(self.Par):
				X_mean = X_mean + (X_cap[i].X_s*X_cap[i].weight)
				self.x[i] = X_cap[i].X_s[0]
				self.y[i] = X_cap[i].X_s[1]
				self.th[i] = X_cap[i].X_s[2]

			Obj_pose = Pose2D()
			Obj_pose.x = X_mean[0]
			Obj_pose.y = X_mean[1]
			Obj_pose.theta = X_mean[2]
			self.pub_pose.publish(Obj_pose)


		elif N == 1:
			y = numpy.empty([2,1])
			x_l = numpy.empty([N,1])
			y_l = numpy.empty([N,1])

			# storing all the positions of the sensors in range
			for i in range(N):
				x_l[i] = msg.readings[i].landmark.x
				y_l[i] = msg.readings[i].landmark.y	

			# CHECKING IF PARTICLES ARE CLOSE TO RANGE AND BEARING
			d = numpy.zeros([self.Par,1])
			e = numpy.zeros([self.Par,1])
			for i in range(self.Par):
				d[i] = math.sqrt( (X_cap[i].X_s[0]-x_l[0])*(X_cap[i].X_s[0]-x_l[0]) + (X_cap[i].X_s[1]-y_l[0])*(X_cap[i].X_s[1]-y_l[0]))
				e[i] = math.atan2(y_l[0]-X_cap[i].X_s[1], x_l[0]-X_cap[i].X_s[0]) - X_cap[i].X_s[2]

			# print 'd',d
			# print 'e',e
			y[0] = msg.readings[0].range
			y[1] = msg.readings[0].bearing

			err_range = numpy.empty([self.Par,1])
			err_bearing = numpy.empty([self.Par,1])

			for i in range(self.Par):
				err_range[i] = y[0] - d[i]
				err_bearing[i] = y[1] - e[i]

			err_r = 0
			err_b = 0
			err_range_norm = numpy.empty([self.Par,1])
			err_bearing_norm = numpy.empty([self.Par,1])
			total_error = numpy.empty([self.Par,1])

			for i in range(self.Par):
				err_r = err_r + err_range[i]
				err_b = err_b + err_bearing[i]

			for i in range(self.Par):
				err_range_norm[i] = err_range[i]/err_r
				err_bearing_norm[i] = err_bearing[i]/err_b

			# TOTAL ERROR PER PARTICLE
			Sum = 0
			for i in range(self.Par):
				total_error[i] = err_range_norm[i] + err_bearing_norm[i]
				X_cap[i].weight = 1.0/total_error[i]
				X_cap[i].weight = numpy.absolute(X_cap[i].weight)
				# print X_cap[i].weight
				Sum = Sum + X_cap[i].weight


			a = numpy.empty([self.Par,1])
			for i in range(self.Par):
				a[i] = Sum

			b = numpy.random.uniform(1,Sum)
			wts = numpy.empty([self.Par,1])
			for i in range(self.Par):
				wts[i] = X_cap[i].weight
			cond = False
			while True:
				s = a
				s = numpy.subtract(s,wts)
				for j in range(self.Par):
					if (s[j] <= b):
						i = i+1
						cond = True
						X_new[i].X_s = X_cap[j].X_s
						X_new[i].weight = X_cap[j].weight
						break
				if cond == True:
					break
					
			if len(X_new)==self.Par:
				X_mean = numpy.empty([3,1])
				for i in range(self.Par):
					X_mean = X_mean + (X_new[i].X_s*X_new[i].weight)
					self.x[i] = X_new[i].X_s[0]
					self.y[i] = X_new[i].X_s[1]
					self.th[i] = X_new[i].X_s[2]

				print X_mean

				Obj_pose = Pose2D()
				Obj_pose.x = X_mean[0]
				Obj_pose.y = X_mean[1]
				Obj_pose.theta = X_mean[2]
				self.pub_pose.publish(Obj_pose)

		elif N == 2:
			y = numpy.empty([4,1])
			x_l = numpy.empty([N,1])
			y_l = numpy.empty([N,1])

			# storing all the positions of the sensors in range
			for i in range(N):
				x_l[i] = msg.readings[i].landmark.x
				y_l[i] = msg.readings[i].landmark.y	

			# CHECKING IF PARTICLES ARE CLOSE TO RANGE AND BEARING
			d = numpy.zeros([self.Par,1])
			e = numpy.zeros([self.Par,1])
			f = numpy.zeros([self.Par,1])
			g = numpy.zeros([self.Par,1])
			for i in range(self.Par):
				d[i] = math.sqrt( (X_cap[i].X_s[0]-x_l[0])*(X_cap[i].X_s[0]-x_l[0]) + (X_cap[i].X_s[1]-y_l[0])*(X_cap[i].X_s[1]-y_l[0]))
				e[i] = math.atan2(y_l[0]-X_cap[i].X_s[1], x_l[0]-X_cap[i].X_s[0]) - X_cap[i].X_s[2]
				f[i] = math.sqrt( (X_cap[i].X_s[0]-x_l[1])*(X_cap[i].X_s[0]-x_l[1]) + (X_cap[i].X_s[1]-y_l[1])*(X_cap[i].X_s[1]-y_l[1]))
				g[i] = math.atan2(y_l[1]-X_cap[i].X_s[1], x_l[1]-X_cap[i].X_s[0]) - X_cap[i].X_s[2]

			# print 'd',d
			# print 'e',e
			y[0] = msg.readings[0].range
			y[1] = msg.readings[0].bearing
			y[2] = msg.readings[1].range
			y[3] = msg.readings[1].bearing

			err_range = numpy.empty([self.Par,1])
			err_bearing = numpy.empty([self.Par,1])
			err_range_l = numpy.empty([self.Par,1])
			err_bearing_l = numpy.empty([self.Par,1])

			for i in range(self.Par):
				err_range[i] = y[0] - d[i]
				err_bearing[i] = y[1] - e[i]
				err_range_l[i] = y[2] - f[i]
				err_bearing_l[i] = y[3] - g[i]

			err_r = 0
			err_b = 0
			err_range_norm = numpy.empty([self.Par,1])
			err_bearing_norm = numpy.empty([self.Par,1])
			total_error = numpy.empty([self.Par,1])

			for i in range(self.Par):
				err_r = err_r + err_range[i] + err_range_l[i] 
				err_b = err_b + err_bearing[i] + err_bearing_l[i]

			for i in range(self.Par):
				err_range_norm[i] = (err_range[i]+err_range_l[i])/err_r
				err_bearing_norm[i] = (err_bearing[i]+err_bearing_l[i])/err_b

			# TOTAL ERROR PER PARTICLE
			Sum = 0
			for i in range(self.Par):
				total_error[i] = err_range_norm[i] + err_bearing_norm[i]
				X_cap[i].weight = 1.0/total_error[i]
				X_cap[i].weight = numpy.absolute(X_cap[i].weight)
				# print X_cap[i].weight
				Sum = Sum + X_cap[i].weight


			a = numpy.empty([self.Par,1])
			for i in range(self.Par):
				a[i] = Sum

			b = numpy.random.uniform(1,Sum)
			wts = numpy.empty([self.Par,1])
			for i in range(self.Par):
				wts[i] = X_cap[i].weight
			i = 0
			cond = False
			while True:
				s = a
				s = numpy.subtract(s,wts)
				for j in range(self.Par):
					if (s[j] <= b):
						i = i+1
						cond = True
						X_new[i].X_s = X_cap[j].X_s
						X_new[i].weight = X_cap[j].weight
						break
				if cond == True:
					break
					
			if len(X_new)==self.Par:
				X_mean = numpy.empty([3,1])
				for i in range(self.Par):
					X_mean = X_mean + (X_new[i].X_s*X_new[i].weight)
					self.x[i] = X_new[i].X_s[0]
					self.y[i] = X_new[i].X_s[1]
					self.th[i] = X_new[i].X_s[2]

				print X_mean

				Obj_pose = Pose2D()
				Obj_pose.x = X_mean[0]
				Obj_pose.y = X_mean[1]
				Obj_pose.theta = X_mean[2]
				self.pub_pose.publish(Obj_pose)

		elif N == 3:
			y = numpy.empty([6,1])
			x_l = numpy.empty([N,1])
			y_l = numpy.empty([N,1])

			# storing all the positions of the sensors in range
			for i in range(N):
				x_l[i] = msg.readings[i].landmark.x
				y_l[i] = msg.readings[i].landmark.y	

			# CHECKING IF PARTICLES ARE CLOSE TO RANGE AND BEARING
			d = numpy.zeros([self.Par,1])
			e = numpy.zeros([self.Par,1])
			f = numpy.zeros([self.Par,1])
			g = numpy.zeros([self.Par,1])
			h = numpy.zeros([self.Par,1])
			k = numpy.zeros([self.Par,1])
			for i in range(self.Par):
				d[i] = math.sqrt( (X_cap[i].X_s[0]-x_l[0])*(X_cap[i].X_s[0]-x_l[0]) + (X_cap[i].X_s[1]-y_l[0])*(X_cap[i].X_s[1]-y_l[0]))
				e[i] = math.atan2(y_l[0]-X_cap[i].X_s[1], x_l[0]-X_cap[i].X_s[0]) - X_cap[i].X_s[2]
				f[i] = math.sqrt( (X_cap[i].X_s[0]-x_l[1])*(X_cap[i].X_s[0]-x_l[1]) + (X_cap[i].X_s[1]-y_l[1])*(X_cap[i].X_s[1]-y_l[1]))
				g[i] = math.atan2(y_l[1]-X_cap[i].X_s[1], x_l[1]-X_cap[i].X_s[0]) - X_cap[i].X_s[2]
				h[i] = math.sqrt( (X_cap[i].X_s[0]-x_l[2])*(X_cap[i].X_s[0]-x_l[2]) + (X_cap[i].X_s[1]-y_l[2])*(X_cap[i].X_s[1]-y_l[2]))
				k[i] = math.atan2(y_l[2]-X_cap[i].X_s[1], x_l[2]-X_cap[i].X_s[0]) - X_cap[i].X_s[2]

			# print 'd',d
			# print 'e',e
			y[0] = msg.readings[0].range
			y[1] = msg.readings[0].bearing
			y[2] = msg.readings[1].range
			y[3] = msg.readings[1].bearing
			y[4] = msg.readings[2].range
			y[5] = msg.readings[2].bearing

			err_range = numpy.empty([self.Par,1])
			err_bearing = numpy.empty([self.Par,1])
			err_range_l = numpy.empty([self.Par,1])
			err_bearing_l = numpy.empty([self.Par,1])
			err_range_ll = numpy.empty([self.Par,1])
			err_bearing_ll = numpy.empty([self.Par,1])

			for i in range(self.Par):
				err_range[i] = y[0] - d[i]
				err_bearing[i] = y[1] - e[i]
				err_range_l[i] = y[2] - f[i]
				err_bearing_l[i] = y[3] - g[i]
				err_range_ll[i] = y[4] - h[i]
				err_bearing_ll[i] = y[5] - k[i]

			err_r = 0
			err_b = 0
			err_range_norm = numpy.empty([self.Par,1])
			err_bearing_norm = numpy.empty([self.Par,1])
			total_error = numpy.empty([self.Par,1])

			for i in range(self.Par):
				err_r = err_r + err_range[i] + err_range_l[i] + err_range_ll[i]
				err_b = err_b + err_bearing[i] + err_bearing_l[i] + err_bearing_ll[i]

			for i in range(self.Par):
				err_range_norm[i] = (err_range[i]+err_range_l[i]+err_range_ll[i])/err_r
				err_bearing_norm[i] = (err_bearing[i]+err_bearing_l[i]+err_bearing_ll[i])/err_b

			# TOTAL ERROR PER PARTICLE
			Sum = 0
			for i in range(self.Par):
				total_error[i] = err_range_norm[i] + err_bearing_norm[i]
				X_cap[i].weight = 1.0/total_error[i]
				X_cap[i].weight = numpy.absolute(X_cap[i].weight)
				Sum = Sum + X_cap[i].weight


			a = numpy.empty([self.Par,1])
			for i in range(self.Par):
				a[i] = Sum

			b = numpy.random.uniform(1,Sum)
			wts = numpy.empty([self.Par,1])
			for i in range(self.Par):
				wts[i] = X_cap[i].weight
			cond = False
			while True:
				s = a
				s = numpy.subtract(s,wts)

				for j in range(self.Par):
					if (s[j] <= b):
						cond = True
						X_new[i].X_s = X_cap[j].X_s
						X_new[i].weight = X_cap[j].weight
						break
				if cond == True:
					break
					
			if len(X_new)==self.Par:
				X_mean = numpy.empty([3,1])
				for i in range(self.Par):
					X_mean = X_mean + (X_new[i].X_s*X_new[i].weight)
					self.x[i] = X_new[i].X_s[0]
					self.y[i] = X_new[i].X_s[1]
					self.th[i] = X_new[i].X_s[2]

				Obj_pose = Pose2D()
				Obj_pose.x = X_mean[0]
				Obj_pose.y = X_mean[1]
				Obj_pose.theta = X_mean[2]
				self.pub_pose.publish(Obj_pose)





if __name__ == '__main__':
	rospy.init_node('estimator_particlefilter', anonymous=True)
	pf = ParticleFilter()
	rospy.spin()