import numpy as np
from single_scattering import *
import collections
    
def get_aop(sun_az, sun_zen_low, sun_zen_high, sun_zen_inter, cam_head_low, cam_head_high, cam_head_inter):
    aop_all = []

    for sun_zen in range(sun_zen_low, sun_zen_high + 1, sun_zen_inter):
        aop_ascending = []
        for cam_head in range(cam_head_low, cam_head_high + 1, cam_head_inter):
            aop_ascending.append(oceanaop(sun_az,sun_zen,cam_head,cam_elev=0,m2=1.33,npart=1.08,mu=3.483) * 360 / np.pi)
        aop_all.append(aop_ascending)
    # change to array
    aop_all = np.asarray(aop_all)
        
    return aop_all

def aop_2_truthMap(sun_az, sun_zen_low, sun_zen_high, sun_zen_inter, cam_head_low, cam_head_high, cam_head_inter):
    # sun_az should be positive
    if sun_az <= 0:
        sun_az *= -1
    # This method of stack only can be used in our experiment
    aop_all_1 = get_aop(-1 * sun_az, sun_zen_low, sun_zen_high, sun_zen_inter, cam_head_low, cam_head_high, cam_head_inter)
    aop_all_2 = get_aop(sun_az, sun_zen_low, sun_zen_high, sun_zen_inter, cam_head_low, cam_head_high, cam_head_inter)
    
    aop_all = np.vstack((aop_all_1, aop_all_2))
    
    # move first col to last col
    first_col = aop_all[:, 0].reshape(-1, 1)
    aop_all = aop_all[:, 1:]
    aop_all = np.hstack((aop_all, first_col))
    
    # get mapping of each ground truth
    change_idx = first_col.shape[0] // 2
    truth_map = collections.defaultdict(list)
    for i in range(aop_all.shape[0]):
        if i >= change_idx:
            truth_map[i] += [sun_zen_low + sun_zen_inter *  (i % change_idx),  sun_az]
        else:
            truth_map[i] += [sun_zen_low + sun_zen_inter *  i,  -1 * sun_az]
    
    return truth_map, aop_all

