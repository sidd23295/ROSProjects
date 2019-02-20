#!/usr/bin/env python

# Assignment 4 - Siddharth Chamarthy
# UNI - sc4253
# my own implementation of numerical IK

import numpy
import sys
import math
import random

from geometry_msgs.msg import *
import moveit_msgs.msg
import moveit_msgs.srv
import rospy
import tf
from urdf_parser_py.urdf import URDF
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import *

def convert_message(M):
	# gathers the translation part from the message
	t = tf.transformations.translation_matrix((M.translation.x,
												  M.translation.y,
												  M.translation.z))
	# gathers the rotation part from the message
	r = tf.transformations.quaternion_matrix((M.rotation.x,
												M.rotation.y,
												M.rotation.z,
												M.rotation.w))
	# gets the net transformation
	M = numpy.dot(t,r)
	return M

def convert_to_message(T):
	t = geometry_msgs.msg.Pose()
	position = tf.transformations.translation_from_matrix(T)
	orientation = tf.transformations.quaternion_from_matrix(T)
	t.position.x = position[0]
	t.position.y = position[1]
	t.position.z = position[2]
	t.orientation.x = orientation[0]
	t.orientation.y = orientation[1]
	t.orientation.z = orientation[2]
	t.orientation.w = orientation[3]        
	return t

# This function converts the given rotation matrix to angle and axis
def rotation_from_matrix(matrix):
	R = numpy.array(matrix, dtype=numpy.float64, copy=False)
	R33 = R[:3, :3]
	l, W = numpy.linalg.eig(R33.T)
	i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
	if not len(i):
		raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
	axis = numpy.real(W[:, i[-1]]).squeeze()
	l, Q = numpy.linalg.eig(R)
	i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
	if not len(i):
		raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
	cosa = (numpy.trace(R33) - 1.0) / 2.0
	if abs(axis[2]) > 1e-8:
		sina = (R[1, 0] + (cosa-1.0)*axis[0]*axis[1]) / axis[2]
	elif abs(axis[1]) > 1e-8:
		sina = (R[0, 2] + (cosa-1.0)*axis[0]*axis[2]) / axis[1]
	else:
		sina = (R[2, 1] + (cosa-1.0)*axis[1]*axis[2]) / axis[0]
	angle = math.atan2(sina, cosa)
	return angle, axis        

# This function calculates the skew matrix of an input vector
def skew_mat(M):
	skew = numpy.zeros((3,3))
	skew[0,1] = -M[2]
	skew[0,2] = M[1]
	skew[1,0] = M[2]
	skew[1,2] = -M[0]
	skew[2,0] = -M[1]
	skew[2,1] = M[0]
	return skew


class MoveArm(object):

	def __init__(self):
		# Loads the robot model
		self.robot = URDF.from_parameter_server()
		
		# These are all the variables used in the program
		self.num_joints = 0
		self.active_num_joints = 0
		self.passive_num_joints = 0
		self.joint_names = []
		self.joint_axes = []
		self.joint_transforms_list = []
		self.x_current = tf.transformations.identity_matrix()
		self.active_joint_axes = []
		self.J = []
		self.q_current = []
		self.active_joint_names = []

		# Gets all the general information about the robot
		self.joint_info()
		rospy.wait_for_service("compute_ik")
		self.ik_service = rospy.ServiceProxy('compute_ik',  moveit_msgs.srv.GetPositionIK)
		print "IK service ready"

		# Wait for validity check service
		rospy.wait_for_service("check_state_validity")
		self.state_valid_service = rospy.ServiceProxy('check_state_validity',  
													  moveit_msgs.srv.GetStateValidity)
		print "State validity service ready"

		self.n_j = 7

		self.names = ["lwr_arm_0_joint","lwr_arm_1_joint","lwr_arm_2_joint","lwr_arm_3_joint","lwr_arm_4_joint","lwr_arm_5_joint","lwr_arm_6_joint"]

		self.current_joint_state = JointState()
		# MoveIt parameter
		self.group_name = "lwr_arm"

		rospy.Subscriber("/joint_states", JointState, self.j_callback)

		rospy.Subscriber("/motion_planning_goal", Transform, self.plan_callback)

		self.pub = rospy.Publisher("/joint_trajectory", trajectory_msgs.msg.JointTrajectory, queue_size = 1)

	def joint_info(self):
		self.num_joints = 0
		self.active_num_joints = 0
		self.passive_num_joints = 0
		self.joint_names = []
		self.active_joint_axes = []
		self.joint_axes = []
		self.active_joint_names = []
		self.J = []
		link = self.robot.get_root()
		while True:
			if link not in self.robot.child_map: 
				break            
			(joint_name, next_link) = self.robot.child_map[link][0]
			# puts the joint object into a variable
			current_joint_object = self.robot.joint_map[joint_name] 
			# appends all the joint names to a list       
			self.joint_names.append(current_joint_object.name)
			# updates the number of joints in a variable
			self.num_joints = self.num_joints + 1
			
			if current_joint_object.type == 'revolute':
				# updates the active joints number in a list
				self.active_num_joints = self.active_num_joints + 1
				# appends all the joint axes to a list
				self.joint_axes.append(current_joint_object.axis)
				# stores the active joint names in a list
				self.active_joint_names.append(current_joint_object.name)
			else:
				# updates the passive joint number in a list
				self.passive_num_joints = self.passive_num_joints + 1
				# stores the passive joint axes to a list
				self.joint_axes.append(current_joint_object.axis)
			# appends the joint objects to a list
			self.J.append(current_joint_object)
			link = next_link

	def forward_kinematics(self, joint_state):
			# list of joint transforms
			joint_transforms = []
			T = tf.transformations.identity_matrix()
			# stores the translation and rotation from world link to first base joint
			t_base = tf.transformations.translation_matrix(self.J[0].origin.xyz)
			r_base = tf.transformations.euler_matrix(self.J[0].origin.rpy[0],self.J[0].origin.rpy[1],self.J[0].origin.rpy[2],'rxyz')
			T = tf.transformations.concatenate_matrices(t_base,r_base)
			# appends the transform to the list
			joint_transforms.append(T)
			# loop for forward kinematics
			for i in range(1,self.num_joints):
			
				if (self.J[i].type == 'fixed'):
					t = tf.transformations.translation_matrix(self.J[i].origin.xyz)
					r = tf.transformations.euler_matrix(self.J[i].origin.rpy[0],self.J[i].origin.rpy[1],self.J[i].origin.rpy[2],'rxyz')
					t_r = tf.transformations.concatenate_matrices(t,r)
				else:
					t = tf.transformations.translation_matrix(self.J[i].origin.xyz)
					r = tf.transformations.euler_matrix(self.J[i].origin.rpy[0],self.J[i].origin.rpy[1],self.J[i].origin.rpy[2],'rxyz')
					t_r = tf.transformations.concatenate_matrices(t,r)
				# checking index of the joint state
				try:
					A = joint_state.name.index(self.joint_names[i-1])
					q = joint_state.position[A]
				except:
					q = 0.0

				if self.J[i-1].type =='revolute':
					ax = self.J[i-1].axis
					R = tf.transformations.quaternion_matrix(tf.transformations.quaternion_about_axis(q, ax))
				else:
					R = tf.transformations.identity_matrix()
				# final transformation from base to each joint transforms 
				T = tf.transformations.concatenate_matrices(T,R,t_r)
				joint_transforms.append(T)

				# calculation of base to end effector transform and storing it as X_current
				if self.J[self.num_joints-1].type == 'revolute':

					# checking index of the joint state
					try:
						A = joint_state.name.index(self.joint_names[self.num_joints-1])
						q = joint_state.position[A]
					except:
						q = 0.0

					# getting the axis of the joint
					axis = self.J[self.num_joints-1].axis
					Q = tf.transformations.quaternion_matrix(tf.transformations.quaternion_about_axis(q, axis))
					X_current = tf.transformations.concatenate_matrices(T,Q)
				else:
					Q = tf.transformations.identity_matrix()
					X_current = tf.transformations.concatenate_matrices(T,Q)
			# returns the base to end effector transform and joint transforms to both the ik_call back function and cc_callback function
			return (X_current, joint_transforms)

	def IK(self, T):

		# getting the desired base to end effector pose from the Transform message of geometry_msgs message
		x_desired = T

		# a list to store the final joint position guess
		final_joint_pos = []
		current_joint_state = JointState()

		# index the right joint names with the correct joint state
		current_joint_state.name = self.active_joint_names

		# initialize a zeros list for the guess 
		q_c_rand = numpy.zeros((self.active_num_joints))

		# iteration variables
		l = 0
		m = 0
		n = 0

		# boolean condition
		found = False

		# main loop for number of tries
		while not found and n <50:
			m = 0
			n = n+1

			# loop for getting random position values for performing inverse kinematics
			for k in range(self.active_num_joints):
				N = random.uniform(-1*math.pi,1*math.pi)
				q_c_rand[k] = N
			q = True

			# loop for performing convergence of delta x which is the displacement between the desired end effector pose and the current end effector pose
			while q == True and m<250:

				# updates the position in the joint state with every new guess
				current_joint_state.position = q_c_rand

				# gets new base to end effector and joint transforms from the forward kinematics using the joint state of inverse kinematics
				x_cur, joint_trans = self.forward_kinematics(current_joint_state)

				L = len(joint_trans)

				# calculation of the end effector to base for calculation purposes
				x_cur_inv = tf.transformations.inverse_matrix(x_cur)

				# getting the difference in the target end effector pose to the current end effector pose
				ee_ee_ik = tf.transformations.concatenate_matrices(x_cur_inv,x_desired)

				# getting the rotational component of the transform
				r_ee_ik = ee_ee_ik[:3,:3]
				A , B = rotation_from_matrix(r_ee_ik)

				# getting the rotational part of delta x for a gain of 1
				r_v_ee_ik = numpy.dot(A,B) 

				# getting the translational part of delta x for a gain of 1
				t_v_ee_ik = tf.transformations.translation_from_matrix(ee_ee_ik)
		
				# list of both the translational and rotational parts of delta x
				del_x_ik = numpy.append(t_v_ee_ik,r_v_ee_ik)
		

				# Jacobian calculation and assembly loop
				Jacobian = numpy.empty((6,0))
				for i in range(L):
					b_j = joint_trans[i]
					j_b = tf.transformations.inverse_matrix(b_j)
					j_ee = numpy.dot(j_b,x_cur)
					ee_j = tf.transformations.inverse_matrix(j_ee)
					R_ee_j = ee_j[:3,:3]
					t_j_ee = tf.transformations.translation_from_matrix(j_ee)
					s = skew_mat(t_j_ee)
					S = numpy.dot(-1*R_ee_j,s)
					A = numpy.append(R_ee_j,S,axis = 1)
					B = numpy.append(numpy.zeros([3,3]),R_ee_j, axis = 1)
					# velocity transformation matrix
					vj = numpy.append(A,B,axis = 0)
					# x axis alignment
					if (self.joint_axes[i] == ([1,0,0])):
						Jacobian = numpy.column_stack((Jacobian,vj[:,3]))
					# y axis alignment
					elif (self.joint_axes[i] == ([0,1,0])):
						Jacobian = numpy.column_stack((Jacobian,vj[:,4]))
					# z axis alignment
					elif (self.joint_axes[i] == ([0,0,1])):
						Jacobian = numpy.column_stack((Jacobian,vj[:,5]))
					# negative x axis alignment
					elif (self.joint_axes[i] == ([-1,0,0])):
						Jacobian = numpy.column_stack((Jacobian,-1*vj[:,3]))
					# negative y axis alignment
					elif (self.joint_axes[i] == ([0,-1,0])):
						Jacobian = numpy.column_stack((Jacobian,-1*vj[:,4]))
					# negative z axis alignment
					elif (self.joint_axes[i] == ([0,0,-1])):
						Jacobian = numpy.column_stack((Jacobian,-1*vj[:,5]))
					# fixed axis ignorance
					elif (self.joint_axes[i] == 'None'):
						continue

				# Calculation of pseudo Jacobian inverse
				J_pseudo_inv_ik = numpy.linalg.pinv(Jacobian)

				# calculation of joint displacement
				del_q = numpy.dot(J_pseudo_inv_ik,del_x_ik)
				# print del_q
				# calculation of new increment in joint value
				q_c_rand = q_c_rand + del_q

				# to check if delta x is within allowable tolerance ; tolerance = 0.01 (can be changed)
				for j in range(len(del_x_ik)):
					if (numpy.linalg.norm(del_x_ik) < 0.0001):
						l = l+1
				if l == len(del_x_ik):
					final_joint_pos = q_c_rand
					if self.is_state_valid(final_joint_pos) == True:
						found = True
					 	for i in range(7):
					 		if final_joint_pos[i]>2*math.pi:
					 			final_joint_pos[i] = numpy.mod(final_joint_pos[i],2*math.pi)
					 		elif final_joint_pos[i]<-2*math.pi:
					 			final_joint_pos[i] = numpy.mod(final_joint_pos[i],-2*math.pi)
					 	if final_joint_pos[i]>math.pi:
					 		final_joint_pos[i] = final_joint_pos[i]-(2*math.pi)
					 	elif final_joint_pos[i]<-1*math.pi:
					 		final_joint_pos[i] = final_joint_pos[i]+(2*math.pi)
					 	return final_joint_pos
				else:
					# iteration count
					m = m+1
					if m == 250:
						if n<20:
							print 'taking new guess'
						q = False
			# allowing maximum tries of 20
			if n == 20:
				print 'reached maximum number of tries please check if the end effector is in the workspace of the robot and try again'
				return 0
					



	def is_state_valid(self, q):
		req = moveit_msgs.srv.GetStateValidityRequest()
		req.group_name = self.group_name
		req.robot_state = moveit_msgs.msg.RobotState()
		req.robot_state.joint_state.name = ["lwr_arm_0_joint",
											"lwr_arm_1_joint",
											"lwr_arm_2_joint",
											"lwr_arm_3_joint",
											"lwr_arm_4_joint",
											"lwr_arm_5_joint",
											"lwr_arm_6_joint"]
		req.robot_state.joint_state.position = q
		req.robot_state.joint_state.velocity = numpy.zeros(7)
		req.robot_state.joint_state.effort = numpy.zeros(7)
		req.robot_state.joint_state.header.stamp = rospy.get_rostime()
		res = self.state_valid_service(req)
		return res.valid

	def get_values(self, data, j_name):
		if j_name not in data.name:
			return
		k = data.name.index(j_name)
		return data.position[k]

	# this function is used to find the vector between two points
	def vector(self, q1, q2):
		q = numpy.subtract(q1,q2)
		return q

	# this function is used to find the unit vector between two points
	def unit_vector_convertor(self, q1, q2):
		v = self.vector(q1, q2)
		u = v / numpy.linalg.norm(v)
		return u

	# this function is used to find the distance between the two points in joint space
	def norm(self, q1, q2):
		v = self.vector(q1, q2)
		d = numpy.linalg.norm(v)
		return d

	# This function does the main collision check in the rrt algorithm
	def collision_check(self, p_close, p_target):
		q_s = [0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05]

		M = []
		v = self.vector(p_target, p_close)
		u = abs(v)
		# gets number of points for a given sample size
		s = numpy.true_divide(u,q_s)
		t = numpy.ceil(s)
		# gets the maximum of the number of points
		n_points = max(t)
		# divides the vector between the close point and the target point by the 
		# total number of points to give step length
		p = numpy.true_divide(v, n_points)
		# gets the maximum length from the closest point
		l = numpy.multiply(p_close, n_points)
		# gets the vector between target and closest point
		m = numpy.subtract(l, p_target)
		# gets base vector from this computation 
		b = numpy.true_divide(m, n_points)
		for i in range(int(n_points)):
			# gets all the points between the target point and close point
			M.append((p*(i+1)+b))
		# checks if all the lists in the given array of lists are not colliding
		for t in M:
			if self.is_state_valid(t) == False:
				return False
		return True

	def close(self, rrt_tree, q):
		d = []
		# Since the rrt is stored in a list of dictionary objects we need to enumerate it as long as there is an object in the rrt_tree
		for k,pos in enumerate(p["position"] for p in rrt_tree):
			# All the distances between the current random q and the nodes in the tree are stored in a list
			d.append(self.norm(pos,q))
		# the index of the smallest element is stored in a variable
		min_index = d.index(min(d))
		# the position of the closest point on the tree is returned
		p_close = rrt_tree[min_index].get("position")
		return min_index, p_close


	def rrt_function(self, s1, s2):
		# creation of a dictionary that stores position and index of the specific source 
		dictionary = {"position": s1, "source_node" : -1}
		# object of the dictionary acts like a list
		rrt = []
		rrt.append(dictionary.copy())
		# copies all the dictionary contents to the rrt tree object
		while True:
			# loop to find random joint angles between -pi and +pi
			q_rand = []
			for k in range(7):
				N = random.uniform(-1*math.pi,math.pi)
				q_rand.append(N)

			# This function call is used to find the closest point in the rrt list to the randomly generated joint angles
			min_d_index, p_close = self.close(rrt,q_rand)

			# This code uses the close point from the rrt and finds a point 0.3 units away from it in the direction of the random point
			p_target = self.unit_vector_convertor(q_rand, p_close)
			p_target = numpy.multiply(p_target,0.3)
			p_target = numpy.add(p_target,p_close)

			# This is the beginning of the main algorithm for the rrt collision check
			# primary check: if there is any obstacle between target point and closest point
			if self.collision_check(p_close,p_target) == True:
				dictionary.update({"position": p_target})
				dictionary.update({"source_node": min_d_index})
				rrt.append(dictionary.copy())
				print len(rrt)
				# Secondary check: if there is any obstacle between target point and goal point
				if self.collision_check(p_target, s2) == True:
					source_node = len(rrt)-1
					dictionary.update({"source_node": source_node})
					dictionary.update({"position": s2})
					rrt.append(dictionary.copy())
					print 'goal has been reached'
					break

		# start retracing path backwards
		q_list = [s2]
		source_node = rrt[-1].get("source_node")

		while True:
			pos_node = rrt[source_node].get("position")
			q_list.insert(0, pos_node)
			if source_node <=0:
				break
			else:
				source_node = rrt[source_node].get("source_node")

		# shortcutting by just checking any two points on the q_list		
		copy = []
		copy.append(q_list[0])
		u = 0
		for u in range(len(q_list)-2):
			if self.collision_check(q_list[u], q_list[u+2]) == False:
				copy.append(q_list[u])
		copy.append(q_list[-1])
		q_list = copy

		# Resampling the path so that continuous motion is assured
		q_new = []
		q_new.append(list(q_list[0]))
		c = 1
		z = 0
		while c<(len(q_list)):
			v = self.vector(q_list[c],q_list[z])
			norm = numpy.linalg.norm(v)
			numb_points = norm/0.5
			ceil_n_p = int(numpy.ceil(numb_points))
			if ceil_n_p == 0:
				q_new.append(q_list[z])
				q_new.append(q_list[c])
			else:
				segment_steps = numpy.divide(v,ceil_n_p)
				j = 0
				# loop for appending the new set of points with steps 
				for j in range(ceil_n_p):
					q_new.append(q_list[z]+((j+1)*segment_steps))
			z = z+1
			c = c+1
		return q_new


	def plan_callback(self, data):
		# everytime command is given this function gets a transform from the message 
		# and grabs the transformation from it
		t_g = convert_message(data)

		# the list of starting joint values taken from the current joint state
		q_st = []
		for t in range(0, self.n_j):
			p = self.get_values(self.current_joint_state, self.names[t])
			q_st.append(p)

		# This function solves the IK for the given message and returns joint values
		q_g = self.IK(t_g)
		print q_g
		if len(q_g) == 0:
			print 'IK failed'
			return
		print 'IK solved'

		# converts the given list of joint angles into an array for use in the RRT function
		q_st = numpy.array(q_st)
		# solves the motion planning problem and returns list of joint values which belong to the trajectory
		q_l = self.rrt_function(q_st, q_g)
		# creates the joint trajectory for list of joint angles
		joint_tra = self.trajectory_creator(q_l)
		if not joint_tra.points:
			print 'failed'
		else:
			# publishes the joint trajectory
			self.pub.publish(joint_tra)
			
	def trajectory_creator(self, path):
		t = trajectory_msgs.msg.JointTrajectory()
		for i in range(0, len(path)):
			p = trajectory_msgs.msg.JointTrajectoryPoint()
			# stores the positions as a list 
			p.positions = list(path[i])
			p.velocities = []
			p.accelerations = []
			t.points.append(p)
		t.joint_names = self.names
		return t

	def j_callback(self, joint_states):
		self.current_joint_state = joint_states

if __name__ == '__main__':
	rospy.init_node('move_arm', anonymous=True)
	ma = MoveArm()
	rospy.spin()

