import pims
import trackpy as tp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import os
def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

def mark_picture(x,y, frame):


    data = [[x, y]]
    df = pd.DataFrame(data, columns = ['x', 'y'])
    plt.xlabel('x ')
    tp.annotate(df, frame,  plot_style={'markersize': 2})
    
def highest_X(raw_pic):

    pic = raw_pic[192:346]
    offset = 0
    pixel = np.arange(len(pic))
 


    offset = offset / 100
    
    pic = pic - offset
        
    try:
        popt,pcov = curve_fit(gaus, pixel, pic)
        
        x0 = int(popt[1])
        fitData = gaus(pixel,*popt)

        peak = fitData[x0]
        
    except:
        peak = -1
    
    return peak
 
    
def highest_Y(pic):
    m = 1000
    offset = 0
    pixel = np.arange(len(pic))
    
    
    for k in range(0, 100):
        offset += pic[k] - 1

    offset = offset / 100
    pic = pic - offset

       
    try:
        popt,pcov = curve_fit(gaus, pixel, pic)
        x0 = int(popt[1])
        delta_x0 = np.diag(pcov)[1]
        error = popt[2]
        fitData = gaus(pixel,*popt)

        peak = fitData[x0]
    except:
        peak = -1
        delta_x0 = 0
        error = 0

    return peak, error
 
def Find_Gauss_Intersect(pic):
    
   
    i = 0
    maximum = 0
    peak = 0
    maxIndexY = 0
    maxIndexX = 0
    tempError = 0
    error = 0
    
    for i in range(216,767):
        peak,tempError = highest_Y(pic[i])

        if(peak > maximum):
           maximum = peak
           error = tempError
    
    transposePic = np.transpose(pic)
    maximum = 0
    
    for i in range(0,767):
        peak = highest_X(transposePic[i])
        
        if(peak > maximum):
            maximum = peak
            maxIndexX = i
        
    #print('x = ', maxIndexX,'y = ', maxIndexY)
    #mark_picture(maxIndexX, maxIndexY, pic)
 
    return maxIndexX, error
            
    
def main():

 
    frequencies = np.zeros(28)
    displacement = np.zeros(28)
    error = np.zeros(28)
    i = 0
    k = 0
    directory = os.path.dirname(os.path.realpath(__file__)) + '/Samples/'
    
    for filename in os.listdir(directory):
        tempStr = ''
        print('i')
        if('Hz' in filename):
            displacement[k],error[k] = Find_Gauss_Intersect(pims.TiffStack('./Samples/' + filename)[0])
            
            i = 0
            while(filename[i] != 'H'):

                tempStr += filename[i]
                i += 1

            frequencies[k] = int(tempStr)
            k += 1

    save = open('Data.txt', 'w')
    for i in range(0,28):
        save.write(str(frequencies[i]) + '\t' + str(displacement[i]) + '\t' + str(error[i]) + '\n')

    save.close()
    plt.errorbar(frequencies, displacement,yerr = error, xerr= 5, fmt = 'none',  label = 'Raw Data')
    plt.xlabel("Frequency of Rotating Mirror(Hz)")
    plt.ylabel("Pixel Position (pixel)")

    coef, cov = np.polyfit(frequencies, displacement, 1, cov = True)
    print(cov)
    newEquation = np.poly1d(coef)
    fitData = newEquation(frequencies)
    plt.plot(frequencies, fitData, label = 'Linear Fit')
    plt.title("Slope = " + str(coef[0]))
    plt.legend()
    print(cov)
    plt.show()
            
def plot():
    data = np.loadtxt('data.txt')
    x = data[:,0]
    y = data[:,1]
    err = data[:,2]

    coef, cov = np.polyfit(x, y, 1, cov = True)
    print(cov)
    newEquation = np.poly1d(coef)
    fitData = newEquation(x)
    plt.plot(x, fitData, label = 'Linear Fit')
    
    plt.errorbar(x, y, yerr = err, xerr = 5, fmt='none', label = 'Raw Data')
    plt.title("Slope = " + str(coef[0]))
    plt.legend()
    plt.xlabel("Frequency of Rotating Mirror(Hz)")
    plt.ylabel("Pixel Position (pixel)")
    plt.show()
            
    
#main()
plot()
