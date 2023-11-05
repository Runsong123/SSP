import torch
from torch_scatter import scatter_mean, scatter_max
import numpy as np
from scipy.spatial import cKDTree


def coordinate2index(x, reso, coord_type='3d'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    '''
    x = (x * reso).long()
    
    index = x[:,  0] + reso * (x[:,  1] + reso * x[:, 2])
    index = index[:, None]
    return index



# if __name__ == "__main__":

    



def main():
    print("heelo")
    data_file_path = "F:/MultiViewSR/new_data/adaptive_more_grid/output_res_1_thingi_adaptive_more_grid/"
    epoch = 39000
    resolution = 80
    grid_scale = 2.0 / resolution
    
    
    
    shape_name = "451676"
    surface_grid =np.loadtxt("%s/%s_grid_%d_pts.txt"%(data_file_path, shape_name, resolution))
    
    pts_shape_dir = "F:/MultiViewSR/new_data/optim_data_debug/thingi/"
    pts_shape = np.loadtxt("%s/%s.txt"%(pts_shape_dir, shape_name))
    ptree = cKDTree(pts_shape[:, :3])
    

    pts_log = np.zeros([pts_shape.shape[0],1])

    for i in range(1):
    

        # uniform_grid_num = 16384//3 //(5 *50)
        # uniform_free_grid_num = 16384//3 //(5 *10)
        # uniform_indice = np.random.choice(surface_grid.shape[0], 16384, True)
        
        
        random_gird = surface_grid
        random_ind = []

        for p in np.array_split(random_gird, 10, axis=0):
            # random_noise = grid_scale * np.random.random_sample(p.shape) - grid_scale / 2
            # d = ptree.query(p + random_noise, 50)
            d = ptree.query(p , 50)
            random_ind.extend(d[1][:, :].reshape(-1))
            
        random_ind = np.asarray(random_ind).reshape(-1).tolist()
        
        for item in random_ind:
            pts_log[item] +=1 

    
    no_sample_number = np.sum(pts_log[:,0]<=1)
    print("no_sample_number", no_sample_number)
    
    pts_info = np.concatenate([pts_shape, pts_log],axis=1)
    # sample_infp = 

    threshhold = 3
    np.savetxt("sample_log_%s_%d.txt"%(shape_name, threshhold), pts_info[pts_log[:,0]>=threshhold])
    np.savetxt("miss_log_%s_%d.txt"%(shape_name, threshhold), pts_info[pts_log[:,0]<threshhold])
    # np.savetxt("surface_log_log.txt",np.concatenate([surface_grid,surface_grid_log],axis=1))
    # np.savetxt("surface_normal_log_log.txt",np.concatenate([surface_normal_grid,surface_normal_log],axis=1))
    # np.savetxt("surface_freespace_log_log.txt",np.concatenate([surface_freespace_grid,surface_freespace_grid_log],axis=1))
    # np.savetxt("surface_freespace_normal_log_log.txt",np.concatenate([surface_freespace_normal_grid,surface_freespace_normal_log],axis=1))

def region_sample():
    print("heelo")
    data_file_path = "/apdcephfs/share_1467498/home/runsongzhu/IGR/data/adaptive_more_grid/output_res_1_thingi_adaptive_more_grid/"
    epoch = 39000
    resolution = 80
    grid_scale = 2.0 / resolution
    
    
    
    shape_name = "451676"
    surface_grid =np.loadtxt("%s/%s_grid_%d_pts.txt"%(data_file_path, shape_name, resolution))
    
    pts_shape_dir = "/apdcephfs/share_1467498/home/runsongzhu/IGR/data/thingi_normalized_data_with_PCA_normal/"
    pts_shape = np.load("%s/%s.npy"%(pts_shape_dir, shape_name))[:,:3]
    ptree = cKDTree(pts_shape[:, :3])
    

    resolution = 80
    pts_log = np.zeros([pts_shape.shape[0],1])
    
    torch_pts_shape = torch.from_numpy(pts_shape)
    index = coordinate2index((torch_pts_shape+torch.ones_like(torch_pts_shape))/2, resolution)
    print(index.max())
    fea_grid = torch_pts_shape.new_zeros(3, resolution**3)

    torch_pts_shape_new = torch_pts_shape.permute(1,0)
    print(torch_pts_shape_new.shape)
    print(index.shape)
    fea_grid = scatter_mean(torch_pts_shape_new, index, out=fea_grid)
    # print((fea_grid.abs().sum(0)>0).sum())
    # print(surface_grid.shape[0])
    print(fea_grid.shape)


    print(index.shape)
    
    

def coordinate2index(x, reso, coord_type='3d'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    '''
    x = (x * reso).long()
    # x = (x * reso).astype(np.long)

    
    index = x[:, 0] + reso * (x[:, 1] + reso * x[:, 2])
    index = index[None,:]
    return index

if __name__ == "__main__":
    region_sample()