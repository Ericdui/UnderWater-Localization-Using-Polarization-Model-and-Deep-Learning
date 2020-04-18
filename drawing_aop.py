import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.ticker import FuncFormatter
from single_scattering import *

def draw_aop(sun_az, sun_zen_low, sun_zen_high, sun_zen_inter, cam_head_low, cam_head_high, cam_head_inter):
    def to_degree_cam_head(temp, position):
        return '%1.0f'%(cam_head_low + cam_head_inter * temp) + '°'
    def to_degree_sun_zen(temp, position):
        return '%1.0f'%(sun_zen_low + sun_zen_inter + sun_zen_inter * (temp - 1)) + '°'

    aop_all = []

    for sun_zen in range(sun_zen_low, sun_zen_high + 1, sun_zen_inter):
        aop_ascending = []
        for cam_head in range(cam_head_low, cam_head_high + 1, cam_head_inter):
            aop_ascending.append(oceanaop(sun_az,sun_zen,cam_head,cam_elev=0,m2=1.33,npart=1.08,mu=3.483) * 360 / np.pi)
        aop_all.append(aop_ascending)
    # change to array
    aop_all = np.asarray(aop_all)

    # set colorbar
    cmap = mpl.cm.winter
    plt.imshow(aop_all, cmap=cmap)
    plt.colorbar()  
    # set label
    plt.xlabel('cam_head')
    plt.ylabel('sun_zen')

    # set ticks
    my_x_ticks = np.arange(0, (cam_head_high - cam_head_low) // cam_head_inter  + 1, 1)
    my_y_ticks = np.arange(0, (sun_zen_high - sun_zen_low) // sun_zen_inter + 1, 1)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    # set unit
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_degree_sun_zen))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_degree_cam_head))

    # set title and save
    if sun_az == 90:
        plt.title('aop of camera towards sun, sun_az=%d°'%90)
        plt.savefig('aop_of_camera_towards_sun.png')
    else:
        plt.title('aop of camera backwords sun, sun_az=%d°'%-90)
        plt.savefig('aop_of_camera_backwords_sun.png')


