import numpy as np
import os


def coordinate2index(x, reso, coord_type='3d'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    '''
    # x = (x * reso).long()
    x = (x * reso).astype(np.long)

    
    index = x[:, 0] + reso * (x[:, 1] + reso * x[:, 2])
    index = index[:]
    return index

def init_grid_index(pts,grid_size,size):
    point_index_all = coordinate2index((pts[:, :3]+np.ones_like(pts[:, :3]))/2 ,size)
        # print(self.point_index_all.min())
        # print(self.point_index_all.max())
    all_value = [[] for i in range(grid_size)]
    print(len(all_value))
    for i in range(pts.shape[0]):
        grid_index = point_index_all[i]
        all_value[grid_index].append(i)
    
    return all_value

if __name__ == "__main__":
    loss_path = "/apdcephfs/share_1467498/datasets/runsongzhu/IGR/all_adaptive_grid_v6_v5_new_sample_v3_fix_clean_v1_pure_sample_fixbug/thingi/exps/451676/2022_02_15_13_13_34/checkpoints/save_midden/"
    output_dir = "origin"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    epoch = 39000
    empty_grid = np.loadtxt("%s/empty_%d.txt"%(loss_path, epoch))
    surface_grid =np.loadtxt("%s/surface_%d.txt"%(loss_path, epoch))
    surface_normal_grid =np.loadtxt("%s/surface_normal_%d.txt"%(loss_path, epoch))
    surface_freespace_grid =np.loadtxt("%s/surface_freespace_%d.txt"%(loss_path, epoch))
    surface_freespace_normal_grid =np.loadtxt("%s/surface_freespace_normal_%d.txt"%(loss_path, epoch))


    data_file_path = "/apdcephfs/share_1467498/home/runsongzhu/IGR/data/thingi_normalized_data_with_PCA_normal/"
    epoch = 39000
    resolution = 80
    grid_scale = 2.0 / resolution
    
    
    
    shape_name = "451676"
    pts =np.load("%s/%s.npy"%(data_file_path, shape_name))[:,:3]

    # surface_grid[:,-1:] = np.log(surface_grid[:,-1:]*10000+1)
    # surface_normal_grid[:,-1:] = np.log(surface_normal_grid[:,-1:]*10000+1)
    # surface_freespace_grid[:,-1:] = np.log(surface_freespace_grid[:,-1:]*10000+1)
    # surface_freespace_normal_grid[:,-1:] = np.log(surface_freespace_normal_grid[:,-1:]*10000+1)

    pts_log = np.zeros([pts.shape[0],1])
    pts_normal_log = np.zeros([pts.shape[0],1])
    surface_grid_log = np.zeros([surface_grid.shape[0],1])
    surface_normal_log = np.zeros([surface_grid.shape[0],1])
    surface_freespace_grid_log = np.zeros([surface_freespace_grid.shape[0],1])
    surface_freespace_normal_log = np.zeros([surface_freespace_grid.shape[0],1])


    grid_index = init_grid_index(pts,surface_grid.shape[0],resolution)

        



    for i in range(10000):
    

        uniform_grid_num = 16384//3 //(5 *50)
        uniform_free_grid_num = 16384//3 //(5 *10)
        uniform_indice = np.random.choice(surface_grid.shape[0], uniform_grid_num, True, p = (surface_grid[:,-1])/np.sum(surface_grid[:,-1]))
        uniform_indice_normal = np.random.choice(surface_grid.shape[0], uniform_grid_num, True, p = surface_normal_grid[:,-1] /np.sum(surface_normal_grid[:,-1] ))
        uniform_indice_freespace = np.random.choice(surface_freespace_grid.shape[0], uniform_free_grid_num, True, p = surface_freespace_grid[:,-1]  /np.sum(surface_freespace_grid[:,-1]))
        uniform_indice_freespace_normal = np.random.choice(surface_freespace_normal_grid.shape[0], uniform_free_grid_num, True, p = surface_freespace_normal_grid[:,-1] /np.sum(surface_freespace_normal_grid[:,-1]))
        
        all_pts_index = [np.array(grid_index[indice]) for indice in uniform_indice]
        all_pts_index = np.concatenate(all_pts_index)
        pts_log[all_pts_index] += 1

        all_pts_index = [np.array(grid_index[indice]) for indice in uniform_indice_normal]
        all_pts_index = np.concatenate(all_pts_index)
        pts_normal_log[all_pts_index] += 1

        surface_grid_log[uniform_indice] +=1
        surface_normal_log[uniform_indice_normal] +=1
        surface_freespace_grid_log[uniform_indice_freespace] +=1
        surface_freespace_normal_log[uniform_indice_freespace_normal] +=1
    
    
    np.savetxt("%s/pts_log_log.txt"%output_dir,np.concatenate([pts,pts_log],axis=1))
    np.savetxt("%s/pts_normal_log_log.txt"%output_dir,np.concatenate([pts,pts_normal_log],axis=1))
    np.savetxt("%s/surface_log_log.txt"%output_dir,np.concatenate([surface_grid,surface_grid_log],axis=1))
    np.savetxt("%s/surface_normal_log_log.txt"%output_dir,np.concatenate([surface_normal_grid,surface_normal_log],axis=1))
    np.savetxt("%s/surface_freespace_log_log.txt"%output_dir,np.concatenate([surface_freespace_grid,surface_freespace_grid_log],axis=1))
    np.savetxt("%s/surface_freespace_normal_log_log.txt"%output_dir,np.concatenate([surface_freespace_normal_grid,surface_freespace_normal_log],axis=1))