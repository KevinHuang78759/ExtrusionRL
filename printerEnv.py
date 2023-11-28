# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 09:20:13 2023

@author: Kevin Huang
"""


import numpy as np
import matplotlib.pyplot as plt

#Class to simulate printer, observations, actions, and rewards
class PrinterEnv:
    def __init__(self, lattice_size = (8,8)):
        #Total Number of points in Lattice
        self.total_points = lattice_size[0]*lattice_size[1]
        #Dimensions of Lattice
        self.num_row = lattice_size[0]
        self.per_row = lattice_size[1]
        
        #Material Allocation: alloate some extra material because we want to
        #use it for correction
        self.mat = self.total_points*1.2
        
        #The number of nodes that have actually recieved material from the nozzle
        self.printed = 0
        
        
    
        
        #To evaluate steps, we compare the result to that of the perfect result.
        #Assume perfect result has a print area of 1 at each node, so we actually
        #need to completely evaluate the FE during initialization.
        
        #Mesh Node Labeling:
        
        #Even numbers are horizontal elements, odd numbers are vertical elements
        
        #Also assume lengths are uniform at the start (all are length 1)
        
        #This potentially results in more odds than evens, or vice versa
        
        #So for later iteration we need to do some magic here...
        #All loops that loop through elements will go through with two passes:
            #First handling a continuous regime where we have the same number
            #of odd and even numbers
            
            #Then, do a separate pass with any unpaired stragglers
        self.last_odd = 2 * (self.total_points - self.per_row) - 1
        self.last_even = 2*(self.total_points - self.num_row - 1)
        self.first = min(self.last_odd, self.last_even)
        self.last = max(self.last_odd, self.last_even)
        
        #Create an FE matrix that we can modify in place later to limit
        #space complexity
        
        #Use FRAME elements, since we test both deflection and elongation
        #Each node has 3 degrees of freedom:x displacement, y displacement, and rotation
        self.FE_K = np.zeros((3*self.total_points,3*self.total_points), dtype = np.float64)
        self.FE_F = np.zeros((3*self.total_points), dtype = np.float64)
        
        #We need a rotation matrices for the vertical elements
        self.rot90 = np.array([[ 0, 1, 0, 0, 0, 0],
                               [-1, 0, 0, 0, 0, 0],
                               [ 0, 0, 1, 0, 0, 0],
                               [ 0, 0, 0, 0, 1, 0],
                               [ 0, 0, 0,-1, 0, 0],
                               [ 0, 0, 0, 0, 0, 1]],dtype = np.int8)
        
        #Store the area printed at each lattice point (deposition)
        #Default value of 1 (perfect print) so that FE matrices can be built in-place
        #Save space complexity
        self.area_printed = np.ones((self.total_points), dtype = np.float64)
        
        #Assume isotropic material and uniform Young's Mod. for simplicity
        
        #Define local stiffness matrices and force vector
        #These matrices are taken from Kevin's submission of HW9 in MAE 4700
        
        #Stiffness for elongation
        self.local_stiff_elo = np.array([[ 1, 0, 0,-1, 0, 0],
                                         [ 0, 0, 0, 0, 0, 0],
                                         [ 0, 0, 0, 0, 0, 0],
                                         [-1, 0, 0, 1, 0, 0],
                                         [ 0, 0, 0, 0, 0, 0],
                                         [ 0, 0, 0, 0, 0, 0]],dtype = np.float64)
        
        #Stiffness for deflection
        self.local_stiff_def = np.array([[ 0, 0, 0, 0, 0, 0],
                                         [ 0,12, 6, 0,-12,6],
                                         [ 0, 6, 4, 0,-6, 2],
                                         [ 0, 0, 0, 0, 0, 0],
                                         [ 0,-12,-6, 0,12,-6],
                                         [ 0, 6, 2, 0,-6, 4]],dtype = np.float64)
        
        #Overall Force Vector
        self.local_F = np.array([0, 1./2, 1./12, 0, 1./2, -1./12])
        
        #Shape functions for result interpolation of forces
        self.axial_shape = np.array([-1, 0, 0, 1, 0, 0])
        self.shear_shape = 8 * np.array([0, 3./2, 3./4, 0, -3./2, 3./4])
        
        #Failure thresholds: computed by looking at the result from a perfect
        #print, then increasing it a bit by some arbitrary tolerance
        
        #Stress Threshold
        self.fail_threshold = 0.045
        
        #Displacement Threshold
        self.def_threshold = 2.85
        
        #Compute and store default loations of all the points
        self.default_locations = np.zeros((self.total_points, 2))
        for i in range(0, self.total_points):
            self.default_locations[i,0] = i % self.per_row
            self.default_locations[i,1] =  int(i/self.per_row)
            
        #Tells us if this simulation has terminated yet
        self.terminated = False
        
        #Current Location of the print nozzle along the toolpath
        self.curloc = 0
    
    #Function to reset all necessary values
    def reset(self):
        #Reset print and nozzle locations
        self.printed = 0
        self.curloc = 0
        
        #Reset all areas - back to one so FE can be built in place
        self.area_printed = np.ones((self.total_points), dtype = np.float64)
        
        #Reset termination and reallocate material
        self.mat = self.total_points*1.2
        self.terminated = False

    
    def step(self, action):
        #Start at 0 reward
        reward = 0.0
        
        #Action = 1 means move to next location
        #Action = 2 means deposit material
        progress = self.printed/self.total_points
        #Values chosen so that we can use exlusive or with 3 to swap between them
        if action == 1:
            #move to next location
            self.curloc += 1
            
            #print some small value if we skip a well: nozzles can't just abruptly stop
            #also avoids FE singularities
            if self.printed < self.curloc:
                #Do not move without having printed first
                self.terminated = True
            else:
                self.create_FE()
                Nf, Ef = self.solve_FE()
                #Total number of elements
                Tf = self.per_row*(self.num_row-1) + (self.num_row) * (self.per_row - 1)
                
                #Scale by progress and against number of points
                reward += (1 + progress) * (self.total_points - Nf + Tf - Ef)/(self.total_points + Tf)
                
                #Terminate on too many failures
                if (Ef+Nf)/(self.total_points + Tf) > 0.04:
                    self.terminated = True
                
            #Terminate if we exit the lattice
            if self.curloc >= self.total_points:
                self.terminated = True
                
            #correct move reward (accumulates to -0.1 if we skip a well)
            #reward += 0.5
            
        else:
            
            #Sample a print value with noise: favor underprints slightly
            print_val = np.random.normal(0.9, 0.3)
            #Cap the value to avoid singularities in FE matrix
            print_val = max(print_val, 0.0001)
            
            #If we have not yet printed here, deposit material
            #Otherwise, increment as a correction attempt
            if self.curloc >= self.printed:
                self.area_printed[self.curloc] = print_val
                self.printed += 1
                self.mat -= print_val
            else:
                self.area_printed[self.curloc] += print_val
                self.mat -= print_val
            
            #Critical Overprinting/Overcorrection
            if self.area_printed[self.curloc] >= 4.0:
                self.terminated = True
            #if we run out of material, terminate
            if self.mat <= 0.0:      
                self.terminated = True

                    
            #print reward : amount under 1 if under 1
            reward += self.area_printed[self.curloc] if self.area_printed[self.curloc] <= 1.0 else 0.0
        
        
        #Now create observation: 
        #Fill at current location
        cur_fill = 0.0 if self.curloc == self.printed or self.curloc >= self.total_points else self.area_printed[self.curloc]
        #Fill at previous location
        prev_fill = 1.0 if self.curloc == 0 else self.area_printed[self.curloc - 1]
        #Current Print Progress
        
        #Fraction of material remaining
        #Put it in a vector to return later
        obs = np.array([prev_fill, cur_fill, progress, self.mat/(1.2*self.total_points)])
        #check termination and failure conditions
        if self.terminated:
            
            #Check if print fully finished
            if self.printed >= self.total_points:
                #If this is the case, add remaining material as raw reward
                self.create_FE()
                Nf, Ef = self.solve_FE()
                #Total number of elements
                Tf = self.per_row*(self.num_row-1) + (self.num_row) * (self.per_row - 1)
                
                #Scale by progress and against number of points
                reward += self.mat + 100 * (self.total_points - Nf + Tf - Ef)/(self.total_points + Tf)
            else:
                #we must have run out of material/pre-terminated/too many failures/critically overprinted
                #no termination reward
                reward = 0.0
                
        
        return reward, obs
        
    
    
    
    def create_FE(self):
        #Reset K and F
        self.FE_K = np.zeros((3*self.total_points,3*self.total_points), dtype = np.float64)
        self.FE_F = np.zeros((3*self.total_points), dtype = np.float64)
        #For the force vetor, we will test a distributed downward load
        p_load = -0.2 
        for i in range(0, self.first + 1):
            #Handle even/odd cases seperately
            if i % 2 == 0:
                #even case
                n1 = i//2 + int(np.floor(i/(2*(self.per_row - 1))))
                n2 = n1 + 1
                
                #Use cross sectional area as average for now
                CSArea = (self.area_printed[n1] + self.area_printed[n2])/2.0
                #Create local stiffness
                lcl_stiff = (CSArea * self.local_stiff_elo) + (CSArea*CSArea*self.local_stiff_def)
                #No rotations needed here, so just add to global stiffness
                self.FE_K[n1*3:n1*3+3,n1*3:n1*3+3] += lcl_stiff[:3,:3]
                self.FE_K[n2*3:n2*3+3,n2*3:n2*3+3] += lcl_stiff[3:,3:]
                self.FE_K[n1*3:n1*3+3,n2*3:n2*3+3] += lcl_stiff[:3,3:]
                self.FE_K[n2*3:n2*3+3,n1*3:n1*3+3] += lcl_stiff[3:,:3]
                
                if i >= self.last_even - 2*(self.per_row-2):
                    #If we are on the top row, we contribute to global force vector
                    lcl_force = p_load * self.local_F
                    self.FE_F[n1*3:n1*3+3] += lcl_force[:3]
                    self.FE_F[n2*3:n2*3+3] += lcl_force[3:]
     
            else:
                #odd case
                n1 = (i-1)//2
                n2 = n1 + self.per_row
                #Use cross sectional area as average for now
                CSArea = (self.area_printed[n1] + self.area_printed[n2])/2.0
                #Create local stiffness
                lcl_stiff = (CSArea * self.local_stiff_elo) + (CSArea*CSArea*self.local_stiff_def)
                #Odd are vertical, rotations are needed
                lcl_stiff = self.rot90.T @ lcl_stiff @ self.rot90
                self.FE_K[n1*3:n1*3+3,n1*3:n1*3+3] += lcl_stiff[:3,:3]
                self.FE_K[n2*3:n2*3+3,n2*3:n2*3+3] += lcl_stiff[3:,3:]
                self.FE_K[n1*3:n1*3+3,n2*3:n2*3+3] += lcl_stiff[:3,3:]
                self.FE_K[n2*3:n2*3+3,n1*3:n1*3+3] += lcl_stiff[3:,:3]
        
        #loop through unmatched labels
        for i in range(self.first + 1, self.last+1, 2):
            #Handle even/odd cases seperately
            if i % 2 == 0:
                #even case
                n1 = i//2 + int(np.floor(i/(2*(self.per_row - 1))))
                n2 = n1 + 1
                
                #Use cross sectional area as average for now
                CSArea = (self.area_printed[n1] + self.area_printed[n2])/2.0
                #Create local stiffness
                lcl_stiff = (CSArea * self.local_stiff_elo) + (CSArea*CSArea*self.local_stiff_def)
                #No rotations needed here, so just add to global stiffness
                self.FE_K[n1*3:n1*3+3,n1*3:n1*3+3] += lcl_stiff[:3,:3]
                self.FE_K[n2*3:n2*3+3,n2*3:n2*3+3] += lcl_stiff[3:,3:]
                self.FE_K[n1*3:n1*3+3,n2*3:n2*3+3] += lcl_stiff[:3,3:]
                self.FE_K[n2*3:n2*3+3,n1*3:n1*3+3] += lcl_stiff[3:,:3]
                
                if i >= self.last_even - 2*(self.per_row-2):
                    #If we are on the top row, we contribute to global force vector
                    lcl_force = p_load * self.local_F
                    self.FE_F[n1*3:n1*3+3] += lcl_force[:3]
                    self.FE_F[n2*3:n2*3+3] += lcl_force[3:]
     
            else:
                #odd case
                
                n1 = (i-1)//2
                n2 = n1 + self.per_row
                #Use cross sectional area as average for now
                CSArea = (self.area_printed[n1] + self.area_printed[n2])/2.0
                #Create local stiffness
                lcl_stiff = (CSArea * self.local_stiff_elo) + (CSArea*CSArea*self.local_stiff_def)
                #Odd are vertical, rotations are needed
                lcl_stiff = self.rot90.T @ lcl_stiff @ self.rot90
                self.FE_K[n1*3:n1*3+3,n1*3:n1*3+3] += lcl_stiff[:3,:3]
                self.FE_K[n2*3:n2*3+3,n2*3:n2*3+3] += lcl_stiff[3:,3:]
                self.FE_K[n1*3:n1*3+3,n2*3:n2*3+3] += lcl_stiff[:3,3:]
                self.FE_K[n2*3:n2*3+3,n1*3:n1*3+3] += lcl_stiff[3:,:3]
        
        #Fix displacement/rotation of bottom row to 0
        rm = []
        for i in range(0, self.per_row):
            rm.append(3*i)
            rm.append(3*i+1)
            rm.append(3*i+2)
        
        self.FE_K = np.delete(self.FE_K, rm, axis = 0)
        self.FE_K = np.delete(self.FE_K, rm, axis = 1)
        self.FE_F = np.delete(self.FE_F, rm)
        
       
                
    def solve_FE(self, display = False):
        
        #Here we solve the finite elements
        sln = np.linalg.solve(self.FE_K, self.FE_F)
        
        #Need to interpret results:
            #Find amount of elements over stress threshold
            #Find number of nodes over displacement threshold
        #Loop through them elements again!
        ele_count = 0
        node_count = 0
        el_list = []
        for i in range(0, self.first + 1):
            #Handle even/odd cases seperately
            if i % 2 == 0:
                #even case
                n1 = i//2 + int(np.floor(i/(2*(self.per_row - 1))))
                n2 = n1 + 1
    
                CSArea = (self.area_printed[n1] + self.area_printed[n2])/2.0
                
                n1sln = np.zeros(3)
                n2sln = np.zeros(3)
                if n1 >= self.per_row:
                    n1sln = sln[3*(n1-self.per_row):3*(n1-self.per_row)+3]
                if n2 >= self.per_row:
                    n2sln = sln[3*(n2-self.per_row):3*(n2-self.per_row)+3] 
                
                lcl_sln = np.concatenate((n1sln, n2sln), axis = 0)
                
                F_ax = (self.axial_shape @ lcl_sln)
                F_sh = -CSArea  * (self.shear_shape @ lcl_sln)
                tag = 0
                if F_ax*F_ax + F_sh*F_sh > self.fail_threshold:
                    ele_count += 1
                    tag = 1
       
                if display:
                    line = [self.default_locations[n1,0] + lcl_sln[0], 
                            self.default_locations[n1,1] + lcl_sln[1],
                            self.default_locations[n2,0] + lcl_sln[3],
                            self.default_locations[n2,1] + lcl_sln[4], tag]
                    el_list.append(line)
                    
            else:
                #odd case             
                n1 = (i-1)//2
                n2 = n1 + self.per_row
                
                CSArea = (self.area_printed[n1] + self.area_printed[n2])/2.0
                n1sln = np.zeros(3)
                n2sln = np.zeros(3)
                if n1 >= self.per_row:
                    n1sln = sln[3*(n1-self.per_row):3*(n1-self.per_row)+3]
                if n2 >= self.per_row:
                    n2sln = sln[3*(n2-self.per_row):3*(n2-self.per_row)+3] 
                
                lcl_sln =  self.rot90 @ np.concatenate((n1sln, n2sln), axis = 0)
                
                F_ax =  (self.axial_shape @ lcl_sln)
                F_sh = -CSArea *  (self.shear_shape.T @ lcl_sln)
                tag = 0
                if F_ax*F_ax + F_sh*F_sh > self.fail_threshold:
                    ele_count += 1
                    tag = 1
                if display:
                    lcl_sln = self.rot90.T @ lcl_sln
                    line = [self.default_locations[n1,0] + lcl_sln[0], 
                            self.default_locations[n1,1] + lcl_sln[1],
                            self.default_locations[n2,0] + lcl_sln[3],
                            self.default_locations[n2,1] + lcl_sln[4], tag]
                    el_list.append(line)
                    
        for i in range(self.first + 1, self.last+1, 2):
            if i % 2 == 0:
                #even case
                n1 = i//2 + int(np.floor(i/(2*(self.per_row - 1))))
                n2 = n1 + 1

                CSArea = (self.area_printed[n1] + self.area_printed[n2])/2.0
                
                n1sln = np.zeros(3)
                n2sln = np.zeros(3)
                if n1 >= self.per_row:
                    n1sln = sln[3*(n1-self.per_row):3*(n1-self.per_row)+3]
                if n2 >= self.per_row:
                    n2sln = sln[3*(n2-self.per_row):3*(n2-self.per_row)+3] 
                
                lcl_sln = np.concatenate((n1sln, n2sln), axis = 0)
                
                F_ax = (self.axial_shape @ lcl_sln)
                F_sh = -CSArea *  (self.shear_shape @ lcl_sln)
                tag = 0
                if F_ax*F_ax + F_sh*F_sh > self.fail_threshold:
                    ele_count += 1
                    tag = 1
                if display:
                    line = [self.default_locations[n1,0] + lcl_sln[0], 
                            self.default_locations[n1,1] + lcl_sln[1],
                            self.default_locations[n2,0] + lcl_sln[3],
                            self.default_locations[n2,1] + lcl_sln[4], tag]
                    el_list.append(line)
                 
            else:
                #odd case             
                n1 = (i-1)//2
                n2 = n1 + self.per_row
                
                CSArea = (self.area_printed[n1] + self.area_printed[n2])/2.0
                n1sln = np.zeros(3)
                n2sln = np.zeros(3)
                if n1 >= self.per_row:
                    n1sln = sln[3*(n1-self.per_row):3*(n1-self.per_row)+3]
                if n2 >= self.per_row:
                    n2sln = sln[3*(n2-self.per_row):3*(n2-self.per_row)+3] 
                
                lcl_sln = self.rot90 @ np.concatenate((n1sln, n2sln), axis = 0)
                
                F_ax =  (self.axial_shape @ lcl_sln)
                F_sh = -CSArea * (self.shear_shape @ lcl_sln)
                tag = 0
                
                if F_ax*F_ax + F_sh*F_sh > self.fail_threshold:
                    ele_count += 1
                    tag = 1
                if display:
                    lcl_sln = self.rot90.T @ lcl_sln
                    line = [self.default_locations[n1,0] + lcl_sln[0], 
                            self.default_locations[n1,1] + lcl_sln[1],
                            self.default_locations[n2,0] + lcl_sln[3],
                            self.default_locations[n2,1] + lcl_sln[4], tag]
                    el_list.append(line)
        
        fp_x = []
        fp_y = []
        max_disp = 0.0
        for i in range(0, self.total_points):
            u1 = 0.0
            u2 = 0.0
            
            if i >= self.per_row:
                u1 = sln[3*(i-self.per_row)]
                u2 = sln[3*(i - self.per_row) + 1]
            max_disp = max(max_disp, u2*u2 + u1*u1)
            if u2*u2 + u1*u1 > self.def_threshold:
                if display:
                    fp_x.append(self.default_locations[i, 0] + u1)
                    fp_y.append(self.default_locations[i, 1] + u2)
                node_count += 1
        
        if display:
            np_x = []
            np_y = []
            for i in range(self.printed, self.total_points):
                u1 = 0.0
                u2 = 0.0
                if i >= self.per_row:
                    u1 = sln[3*(i-self.per_row)]
                    u2 = sln[3*(i - self.per_row) + 1]
                np_x.append(self.default_locations[i, 0] + u1)
                np_y.append(self.default_locations[i, 1] + u2)
                
            plt.figure(0, dpi = 600)
            for lst in el_list:
                color_p = 'b'
                if lst[4] == 1:
                    color_p = 'r'
    
                plt.plot([lst[0], lst[2]], [lst[1], lst[3]], color = color_p)
            plt.plot(fp_x, fp_y, 'ro')
            plt.plot(np_x, np_y, 'gx')
                
            plt.show()
            
        return node_count, ele_count


#Run and display the no correction policy
def disp_basic_pol():
    P = PrinterEnv(lattice_size = (10, 10))
    act = 2
    while not P.terminated:
        P.step(act)
        act ^= 3
    P.create_FE()
    P.solve_FE(display = True)

disp_basic_pol()

#Run and display the Zealous Correction Policy
def disp_correction_pol():
    P = PrinterEnv(lattice_size = (10, 10))
    act = 2
    while not P.terminated:
        _, o = P.step(act)
        if o[1] < 1.0:
            act = 2
        else:
            act = 1
    P.create_FE()
    P.solve_FE(display = True)

disp_correction_pol()


def eval_basic_pol(episodes = 5):
    P = PrinterEnv(lattice_size = (10, 10))
    reward = 0.0
    for _ in range(0, episodes):
        P.reset()
        this_rew = 0.0
        act = 2
        while not P.terminated:
            r, o = P.step(act)
            this_rew += r
            act ^= 3

        
        reward += this_rew
    return reward/episodes

#Evaluate Rewards of Zealous Correction over 5 episodes
def basic_correction_pol(episodes = 5):
    P = PrinterEnv(lattice_size = (10, 10))
    reward = 0.0
    for _ in range(0, episodes):
        P.reset()
        this_rew = 0.0
        act = 2
        while not P.terminated:
            r, o = P.step(act)
            this_rew += r
            if o[1] < 1.0:
                act = 2
            else:
                act = 1
        
        reward += this_rew
    return reward/episodes

#Evaluates Rewards of a policy based on a given weight vector over 5 episodes
def eval_with_weights(w, episodes = 3):
    
    P = PrinterEnv(lattice_size = (10, 10))
    reward = 0.0
    for _ in range(0, episodes):
        P.reset()
        this_rew = 0.0
        #always start with a print step
        act = 2
        while not P.terminated:
            r, o = P.step(act)
            this_rew += r
            local_fills = w[0] * o[0] + (1 - w[0]) * o[1]
            global_condition = w[1] * o[2] + (1 - w[1]) * o[3]
            
            #Heurustic
            if local_fills > w[2]:
                act ^= 3
            else:
                if global_condition < w[3]:
                    P.terminated = True
                else:
                    act = 2  
        
        reward += this_rew
 
    return reward/episodes

R1 = eval_basic_pol()
print('no correction: ' + str(R1))
R2 = basic_correction_pol()
print('full correctionL: ' + str(R2))

#This doesn't work for some reason
def threaded_eval(thread_id, w, R_store):
    
    eval_w = np.clip(w, 0.0001, 1.0)
    R_store[thread_id] = eval_with_weights(eval_w)
    
    
    
def train(iterations = 10, buffer_size = 16, acceptance = 4):
    max_w = 1.0
    min_w = 0.0001
    
    #generate starting buffer
    BUFFER = np.random.uniform(min_w, max_w, (buffer_size, 4))
    REWARD = np.zeros(buffer_size)
    r_track = np.zeros(iterations)
    for i in range(iterations):
        #first evaluate each weight 
        
        #Figure out multithreading later
        for j in range(buffer_size):
            threaded_eval(j, BUFFER[j], REWARD)
        
        
        #Now find the top few rewards
        sort_id = REWARD.argsort()
        sort_id = sort_id[::-1]
        keep_id = sort_id[:acceptance]
        #Save mean of the accepted rewards
        r_track[i] = np.mean(REWARD[keep_id ])
        print(REWARD[sort_id])
        #compute mean and std of kept vectors
        means = np.zeros(4)
        stds = np.zeros(4)
        
        for j in range(acceptance):
            w = BUFFER[keep_id[j]]
            means += w
        means /= acceptance
        for j in range(acceptance):
            w = BUFFER[keep_id[j]]
            stds += (w - means) ** 2
        stds /= acceptance
        
        #repopulate buffer with more samples from this distribution
        stds = np.sqrt(stds)
        BUFFER[:acceptance] = BUFFER[keep_id]
        for j in range(acceptance, buffer_size):
            
            new_w = np.random.normal(means, stds)
            BUFFER[j] = new_w
            
    plt.figure(0, dpi = 600)
    plt.plot(r_track)
    plt.show()
    return BUFFER[0]

trained = train()
print(trained)

#Old trained value
#trained_w = np.array([0.5308854, 0.70214601, 1.66305361, 0.31729649])
#New Trained Value
trained_w = np.array([0.496, 0.199, 0.777, 0.001])
trained_w = trained
def disp_weighted_pol(w = trained):
    P = PrinterEnv(lattice_size = (10, 10))
    P.reset()
    #always start with a print step
    act = 2
    while not P.terminated:
        _, o = P.step(act)
      
        local_fills = w[0] * o[0] + (1 - w[0]) * o[1]
        global_condition = w[1] * o[2] + (1 - w[1]) * o[3]
        
        #Heurustic
        if local_fills > w[2]:
            act ^= 3
        else:
            if global_condition < w[3]:
                P.terminated = True
            else:
                act = 2  
    P.create_FE()
    P.solve_FE(display = True)
    
disp_weighted_pol()