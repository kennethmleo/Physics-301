import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

def color_output(wavelength, P, R, Sr, Sg, Sb): 
    """
    Input:
    P - light source spectral power distribution
    R - reflectance of the object
    S - camera spectral sensitivity (red, green, and blue channels)
    
    Output:
    C - camera output [R,G,B]
    """
    
    P = P/P.max()
    R = R/R.max()
    Sr = Sr/Sr.max()
    Sg = Sg/Sg.max()
    Sb = Sb/Sb.max()
    
    
    Cr = sum(P * R * Sr) / sum(P * Sr)
    Cg = sum(P * R * Sg) / sum(P * Sg)
    Cb = sum(P * R * Sb) / sum(P * Sb)
    
    C = [Cr, Cg, Cb]
    
    return C

def visualize_color(intensity, color):
    output = np.zeros([10,10,3])
    output[:,:,0] = int(color[0]*intensity)
    output[:,:,1] = int(color[1]*intensity)
    output[:,:,2] = int(color[2]*intensity)
    output = output.astype(int)
    
    plt.imshow(output)
    plt.axis('off')
    plt.title('RGB: (' + str(int(color[0]*intensity)) + ',' + str(int(color[1]*intensity)) + ',' + str(int(color[2]*intensity)) + ')', fontsize=15)
    plt.show()
    
def recreate_image(df_camera, df_object, df_light_source, intensity):
    '''
    df_camera should have columns with names 'wavelength', 'red', 'green', and 'blue'
    df_object should have columns with names 'wavelength', and reflectance values
    df_light_source should have columns with 'wavelength' and 'illumination'
    title is the name of the object
    '''
    N = 24 #number of macbeth colors
    macbeth = np.zeros([N,N,3])
    colors = []
    
    for i in [i for i in df_object.columns][1:]:
        wavelength = np.copy(df_object.wavelength)
        light_source = np.interp(wavelength, df_light_source.wavelength, df_light_source.illumination)
        camera_red = np.interp(wavelength, df_camera.wavelength, df_camera.red)
        camera_green = np.interp(wavelength, df_camera.wavelength, df_camera.green)
        camera_blue = np.interp(wavelength, df_camera.wavelength, df_camera.blue)

        color = color_output(wavelength, light_source, df_object[i], camera_red, camera_green, camera_blue)
        colors.append(color)
    
    colors = np.array(colors)
    colors_df = pd.DataFrame(colors, columns = ['red', 'green', 'blue'])
    
    macbeth[:,:,0] = colors[:,0] * intensity
    macbeth[:,:,1] = colors[:,1] * intensity
    macbeth[:,:,2] = colors[:,2] * intensity

    macbeth = macbeth.astype(int)

    array1 = np.array([macbeth[0][0:6]])
    array2 = np.array([macbeth[0][6:12]])
    array3 = np.array([macbeth[0][12:18]])
    array4 = np.array([macbeth[0][18:24]])

    array_camera = np.vstack((array1,array2, array3, array4))
    array_camera[array_camera > 255] = 255
        
    return array_camera

def recreate_macbeth(array, intensity):
    #plt.figure(figsize=[9,6])
    plt.imshow(array)
    for i in range(5):
        plt.axhline(y=-0.5+i, color='black', linewidth = 5)
    for j in range(7):
        plt.axvline(x=-0.5+j, color='black', linewidth = 5)

    plt.axhline(y=-0.5, color='black', linewidth = 10)
    plt.axhline(y=3.5, color='black', linewidth = 10)
    plt.axvline(x=-0.5, color='black', linewidth = 10)
    plt.axvline(x=5.5, color='black', linewidth = 10)
    plt.axis('off')