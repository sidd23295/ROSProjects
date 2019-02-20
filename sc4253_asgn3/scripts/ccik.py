#!/usr/bin/env python

# Assignment 3 - Siddharth Chamarthy
# UNI - sc4253

import random
import math
import numpy
from geometry_msgs.msg import *
import rospy
import tf
import tf.msg
from urdf_parser_py.urdf import URDF
from sensor_msgs.msg import JointState
from cartesian_control.msg import CartesianCommand




# This function converts message from the x_target to a transformation that can be used in the cc_callback
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

# This is the class that holds all functions
class CartesianControl(object):

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
		
		# This is where we'll hold the most recent joint angle information we receive on the topic
		self.current_joint_state = JointState()

		# Subscribers for the different topics
		rospy.Subscriber("/joint_states", JointState, self.j_callback)
		rospy.Subscriber("/cartesian_command", CartesianCommand, self.cc_callback)
		rospy.Subscriber("/ik_command", geometry_msgs.msg.Transform, self.ik_callback)

		# Publishers for the different outputs
		self.pub_vel = rospy.Publisher("/joint_velocities", JointState, queue_size = 10)
		self.pub_ik = rospy.Publisher("/joint_command", JointState, queue_size = 10)

	# The actual callback only stores the received information and calls forward kinematics
	def j_callback(self, joint_state):
		# stores the current joint values in the object variable
		self.current_joint_state = joint_state

		# stores the current joint state position in q_current
		self.q_current = joint_state.position


	# This function gets the joint information from the URDF
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



	def cc_callback(self, data):

		# calls the forward_kinematics to get the current end effector position and all the joint transforms
		self.x_current,self.joint_transforms_list= self.forward_kinematics(self.current_joint_state)
		joint_state = JointState()

		n = len(self.joint_transforms_list)

		objective = 'False'

		# gets the secondary objective boolean input from the message
		objective = data.secondary_objective

		# gets the q0_target position from the message 
		q_desired = data.q0_target

		# gets the desired base to end effector position from the message
		x_target = data.x_target
		
		# T_target is basically the base to desired end effector transformation matrix
		T_target = convert_message(x_target)
		
		# gets the current base to end effector transform from the forward kinematics
		b_ee_current = self.x_current
		
		# list to store the joint velocities
		j_v = numpy.zeros(n)
		
		# finds the inverse of matrix for jacobian calculations
		ee_b_current = tf.transformations.inverse_matrix(b_ee_current)
		
		# ee_ee gives the distance between the old end effector position and the desired end effector position
		ee_ee = tf.transformations.concatenate_matrices(ee_b_current,T_target)
		
		# translational gain for velocity calculation
		t_gain = 1
		r_gain = 1

		# getting the translational velocity of the end effector with translational gain control = 1
		# getting the translation and rotation part of the ee_ee transform
		v_t_ee = tf.transformations.translation_from_matrix(ee_ee)*t_gain
		r_ee = ee_ee[:3,:3]

		# getting axis and angle from the rotation matrix configuration
		angle, ax = rotation_from_matrix(r_ee)

		# calculation of rotational velocity of end effector with rotational gain control = 1
		v_r_ee = numpy.dot(angle,ax)*r_gain

		# calculation of the normalized values of the coressponding translational and rotational velocities
		v_r_norm = numpy.linalg.norm(v_r_ee)
		v_t_norm = numpy.linalg.norm(v_t_ee)

		for i in range(0,3):

			# scaling of the translational velocities to less than 0.1 and greater than -0.1
			if(v_t_ee[i] > 0.1):
				v_t_ee = v_t_ee/v_t_norm
			elif(v_t_ee[i] < -0.1):
				v_t_ee = v_t_ee/v_t_norm
			else:
				v_t_ee[i] = v_t_ee[i]

			# scaling of the rotational velocities to less than 1 and greater than -1
			if (v_r_ee[i] > 1):
				v_r_ee = v_r_ee/v_r_norm
			elif(v_r_ee[i] < -1):
				v_r_ee = v_r_ee/v_r_norm
			else:
				v_r_ee[i] = v_r_ee[i]

		# new list containing both the rotational and translational velocities 
		del_x = numpy.append(v_t_ee,v_r_ee)

		# proportional control gain for del_x
		p_gain = 2

		# list of velocities (x_dot)
		V = p_gain * del_x

		# Calculation and assembly of the Jacobian
		Jacobian = numpy.empty((6,0))
		for i in range(n):
			# formation of the velocity transformation matrix
			b_j = self.joint_transforms_list[i]
			j_b = tf.transformations.inverse_matrix(b_j)
			j_ee = numpy.dot(j_b,b_ee_current)
			ee_j = tf.transformations.inverse_matrix(j_ee)
			R_ee_j = ee_j[:3,:3]
			t_j_ee = tf.transformations.translation_from_matrix(j_ee)
			s = skew_mat(t_j_ee)
			S = numpy.dot(-1*R_ee_j,s)
			A = numpy.append(R_ee_j,S,axis = 1)
			B = numpy.append(numpy.zeros([3,3]),R_ee_j, axis = 1)
			# vj is the velocity transformation matrix
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

		# calculation of pseudo inverse of the Jacobian in safe mode
		J_pseudo_inv = numpy.linalg.pinv(Jacobian, rcond = 1e-2)
		
		# calculation of unscaled joint velocities
		j_v = numpy.dot(J_pseudo_inv,V)
		
		# Checking for secondary objective
		if objective == True:
			# Jacobian in unsafe mode
			J_pseudo_inv = numpy.linalg.pinv(Jacobian, rcond = 0)


			U = numpy.array([q_desired - self.q_current[0]])
			U = U*numpy.identity(self.active_num_joints)
			V_J = numpy.identity(self.active_num_joints) - numpy.dot(J_pseudo_inv,Jacobian)
			
			# projecting the velocity into the nullspace of V
			j_v_sec = numpy.dot(V_J,U[:,0])

			# calculation of new joint velocities
			j_v = numpy.dot(J_pseudo_inv,V) + j_v_sec

		# calculating the norm of the joint velocity vector
		j_v_norm = numpy.linalg.norm(j_v)

		# scaling the joint velocities to 1 or -1 depending on the output
		for j in range(0,self.active_num_joints):
			if (j_v[j] > 1):
				j_v = j_v/j_v_norm
			elif (j_v[j]<-1):
				j_v = j_v/j_v_norm
			else:
				j_v[j] = j_v[j]

		# publishing the new scaled joint velocities to the joint state with appropriate joint names
		joint_state.name = self.active_joint_names
		joint_state.velocity = j_v
		self.pub_vel.publish(joint_state)


	# call back for inverse kinematics
	def ik_callback(self, data):

		# getting the desired base to end effector pose from the Transform message of geometry_msgs message
		x_desired = convert_message(data)

		# a list to store the final joint position guess
		final_joint_pos = []
		joint_state_ik = JointState()

		# index the right joint names with the correct joint state
		joint_state_ik.name = self.active_joint_names

		# initialize a zeros list for the guess 
		q_c_rand = numpy.zeros((self.active_num_joints))

		# iteration variables
		l = 0
		m = 0
		n = 0

		# boolean condition
		found = False

		# main loop for number of tries
		while not found and n <20:
			m = 0
			n = n+1

			# loop for getting random position values for performing inverse kinematics
			for k in range(self.active_num_joints):
				N = random.uniform(0,2*math.pi)
				q_c_rand[k] = N
			q = True

			# loop for performing convergence of delta x which is the displacement between the desired end effector pose and the current end effector pose
			while q == True and m<250:

				# updates the position in the joint state with every new guess
				joint_state_ik.position = q_c_rand

				# gets new base to end effector and joint transforms from the forward kinematics using the joint state of inverse kinematics
				x_cur, joint_trans = self.forward_kinematics(joint_state_ik)

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

				# calculation of new increment in joint value
				q_c_rand = q_c_rand + del_q

				# to check if delta x is within allowable tolerance ; tolerance = 0.01 (can be changed)
				for j in range(len(del_x_ik)):
					if (numpy.linalg.norm(del_x_ik) < 0.01):
						l = l+1
				if l == len(del_x_ik):
					final_joint_pos = q_c_rand
					print 'found solution at the', n,'random position guess and ',m, 'convergence iterations'
					found = True
					joint_state_ik.name = self.active_joint_names	
					joint_state_ik.position = final_joint_pos
					print 'the final joint positions are:\n',final_joint_pos
					self.pub_ik.publish(joint_state_ik)
					break

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
					
if __name__ == '__main__':
	rospy.init_node('ccik', anonymous=True)
	
	# creating a class object
	ccik = CartesianControl()
	rospy.spin()
	

