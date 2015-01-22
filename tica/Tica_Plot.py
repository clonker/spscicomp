import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt

class TicaPlot:
    
    def elbowPlot(self, i_eigenVal):

        if i_eigenVal.shape[0] > 1:
            scale = []
            i = 0
    
            while i < i_eigenVal.shape[0]:
                scale.append(i+1)
                i = i+1
        
            plt.plot(scale, i_eigenVal, marker="o")
            plt.show()

        else:
            print("You only have one eigenvalue")
        
#----------------------------------------------------------------------------   

    def scatterPlot(self, i_comp1, i_comp2):

        if i_comp1.shape[0] == i_comp2.shape[0]:

            plt.plot(i_comp1, i_comp2, linestyle="None", marker="o")
            plt.show()

        else:
            print("Vector dimensions must be the same")
#-----------------------------------------------------------------------------
                  
