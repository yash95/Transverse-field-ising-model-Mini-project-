####################################################################
##                   Checking New Code for Plotting              ###
####################################################################

import numpy as np
#import scipy.linalg as la
#import itertools
import math
from numpy import linalg as LA  # For e.values without calculating e.vectors
import matplotlib.pyplot as plt
import pandas as pd


####   N= int(input("Enter the N: "))
####   g= float(input("Enter g : "))


PI = math.pi

sigma_x = np.array([[0,1],[1,0]])
sigma_z = np.array([[1,0],[0,-1]])
I = np.array([[1,0],[0,1]])

#########################################################################
#########      TO OBTAIN  ISING HAMILTONIAN  ############################

#  This funciton gives the Ising Hamiltonian for given N and g

def find_Ising_H(N,g):
    # Transverse Part.
    i=1
    while i<=N:
        if i==1:
            t=sigma_z
        else :
            t = I
        j=2
        while j<=N:
            if i==j:
                t=np.kron(sigma_z,t)
            else:
                t=np.kron(I,t)
            j+=1
        if i==1:
            transverse_total = t
        else:
            transverse_total = transverse_total + t
        i+=1
  # Interaction part.    
    i=1
    while i<=N:
        if i==1 or i==N:
           t=sigma_x
        else:
           t = I
        j=2
        while j<=N:
            if i==j or j==i+1:
                t=np.kron(sigma_x,t)
            else:
                t=np.kron(I,t)
            j+=1
        if i==1:
            interaction = t
        else:
            interaction= interaction + t
        i+=1
    
    H_ising = g*transverse_total + interaction
    return(H_ising)
#####################################################################################


def energy_levels(A):
    eigen = np.around(LA.eigvals(A).real,6)  # This find only e.values...not eigen vectors.
    eigen1 = np.sort(eigen)
    return(eigen1)

################################################################################################



##########################################
#####      INPUT PART FOR g AND N   ######
##########################################


## HERE: The desired values of g is created as an array
g_values = []
for i in range(0,32,2):
    g_values.append(i*0.1)


N_values = [6, 8]



## FROM HERE THE ENERGY VALUES ARE CALCULATED FOR THE GIVEN INPUTS.

Egs_values = []                    # Ground state energies are stored here.
first_excited_values = []          # First excited state energies are stored here.
second_excited_values = []         # Second excited state energies are stored here.

for N in N_values: 
    Egs_N_const = []               # We will fill Egs for const N in this array.
    first_excited_N_const = []     # We will fill first excited state energies for const N in this array.
    second_excited_N_const = []    # We will fill second excited state energies for const N in this array.

    for g in g_values:
        Es = energy_levels( find_Ising_H(N,g) )
        Egs_N_const.append(Es[0])
        first_excited_N_const.append(Es[1])
        second_excited_N_const.append(Es[2])
    Egs_values.append(Egs_N_const)                         # here we added the arrays for ground state 
    first_excited_values.append(first_excited_N_const)      # here we added the arrays for First excited state energies
    second_excited_values.append(second_excited_N_const)      # here we added the arrays for Second excited state energies


print("g values: ", g_values)
print()
print("Ground State Energies: \n", Egs_values)
print()
print("First Excited State Energies: \n", first_excited_values)
print()
print("Second Excited State Energies: \n", second_excited_values)
print()

##    We need to plot the ground state energies, first excited state energies,
##     and second excited state energies as function of g. (in the back of our mind, for different N)
##     For that, we shall use subplot for these each levels. 
##         (For different N, when it is plotted, one plot should contain one energy level for different N)

## Note that for different values of N, energy arrays are "array of arrays".


#   For manually typing each case:


color = 'cgbrmykcgbrmyk'
## The following part plots the desired curves.

##   Plotting Ground State Energies

for i in  range(0,len(N_values)):
    C = str(color[i])
    N_lab = 'N = '+ str(N_values[i])
        
    plt.plot(g_values,Egs_values[i], marker='o', color=C, label = str(N_lab)  )
    plt.legend(loc= "upper right", prop={'size': 6})
    plt.title("Ground State Energies for Various g values")
plt.show()

    
##  Plotting First Excited State Energies    

for i in  range(0,len(N_values)):
    C = str(color[i])
    N_lab = 'N = '+ str(N_values[i])
    
    plt.plot(g_values,first_excited_values[i], marker='*',color=C, label = str(N_lab) )
    plt.legend(loc= "upper right", prop={'size': 6})
    plt.title("First Excited State Energies for Various g values")
plt.show()

    
    
## Plotting Second Excited State Energies    

for i in  range(0,len(N_values)):
    C = str(color[i])
    N_lab = 'N = '+ str(N_values[i])
    
    plt.plot(g_values,second_excited_values[i], marker ='d', color=C, label = str(N_lab) )
    plt.legend(loc= "upper right", prop={'size': 6})
    plt.title("Second Excited State Energies for Various g values")
plt.show()
    

for i in range(0, len(N_values)):
    df = pd.DataFrame({'g': g_values,'Egs': Egs_values[i],'First Excited': first_excited_values[i], 'Second Excited': second_excited_values[i]})
    writer = pd.ExcelWriter('Ising_Data_For_N = '+ str(N_values[i]),engine = 'xlsxwriter')
    df.to_excel(writer, sheet_name = 'sheet')
    writer.save()


## Plotting Energy gap

for i in  range(0,len(N_values)):
    C = str(color[i])
    N_lab = 'N = '+ str(N_values[i])
    
    plt.plot(g_values,second_excited_values[i]-Egs_values[i], marker ='d', color=C, label = str(N_lab) )
    plt.legend(loc= "upper right", prop={'size': 6})
    plt.title("Energy gap for Various g values")
plt.show()
    



