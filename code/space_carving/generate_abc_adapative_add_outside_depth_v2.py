import numpy as np
import numpy as np
from numpy.core import shape_base
from numpy.lib.npyio import genfromtxt
import open3d as o3d
import copy
import glob
import os
from scipy.spatial import cKDTree
import sys
import scipy.spatial as spatial

def normalize_input(data):
    
    print("data shape:",data.shape)
    data = data - np.mean(data,axis=0,keepdims=True)
    max_dis = np.linalg.norm(data,ord=2,axis=1).max() * 1.1
    data = data/max_dis
    return data


def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def calculate_density(data):

    sigma_set = []
    sigma_set_minumum = []
    ptree = cKDTree(data[:,:3])
    # numpy_data = data[:,:3]
    ptree = ptree
    for p in np.array_split(data[:,:3], 100, axis=0):
        d = ptree.query(p, 50 + 1)
        sigma_set.append(d[0][:, -1])
        sigma_set_minumum.append(d[0][:, 5+1])

    sigmas = np.concatenate(sigma_set)
    sigma_set_minumum = np.concatenate(sigma_set_minumum)
    return sigmas,sigma_set_minumum,ptree




def getPCA(data):
    data_new = data - np.mean(data,axis=0)
    c = np.dot(data_new.T,data_new)/(data_new.shape[0]-1)
    eig_vals,eig_vecs = np.linalg.eig(c)
    if (eig_vals<0).sum()>0:
        print("==bug=======")
    min_index = np.argmin(np.abs(eig_vals))
    return eig_vecs[:,min_index]
    # return 

def getPCA_normal(data,neighbor_size):
    data = normalize_input(data)
    sys.setrecursionlimit(int(max(1000, round(data.shape[0]/10)))) # otherwise KDTree construction may run out of recursions
    # 构建Kdtree
    kdtree = spatial.cKDTree(data, 10)
    normal_res = np.zeros([data.shape[0],3])
    for i in range(data.shape[0]):
        ind_local_patch = kdtree.query(data[i],k=neighbor_size)[1]
        normal_res[i,:] = getPCA(data[ind_local_patch])
    return np.concatenate([data,normal_res],axis=1)




def get_grid_depth(grid):
    grid_size = grid.shape[0]
    grid_depth = np.zeros([grid_size, grid_size, grid_size],dtype=np.int64)-1
    visited =np.zeros([grid_size, grid_size, grid_size],dtype=bool)
    cur_depth = 0
    search_list=[]
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                if grid[i,j,k]>=1:
                    search_list.append([i,j,k])
                    grid_depth[i,j,k]= cur_depth
                    visited[i,j,k]=True
    
    # step = 0 
    directions = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    for direction in directions:
        print(direction)
    
    directions_2 = []
    for ix in [-1,0,1]:
        for iy in [-1,0,1]:
            for iz in [-1,0,1]:
                directions_2.append([ix,iy,iz]) 
    # print("==============",len(search_list))
    while(len(search_list)>0):
        new_search_list = []
        for cur_pos in search_list:
            # cur_pos = search_list[step]
            # print(cur_pos)
            # outside_grid[cur_pos[0],cur_pos[1],cur_pos[2]] = True
            for direction in directions:
                # new_pos = [0,0,0]
                new_pos = cur_pos[0] + direction[0], cur_pos[1] + direction[1] , cur_pos[2] + direction[2]
                # for i in range(3):
                #     new_pos[i] = cur_pos[i] + direction[i]
                # print(new_pos)
                # print(new_pos)
                if noValid(new_pos,(grid_size, grid_size, grid_size)):
                    continue
                if visited[new_pos[0],new_pos[1],new_pos[2]]:
                    continue
                
                visited[new_pos[0],new_pos[1],new_pos[2]] = True
                # print("==========")
                new_search_list.append([new_pos[0],new_pos[1],new_pos[2]])
                grid_depth[new_pos[0],new_pos[1],new_pos[2]] = cur_depth+1
        cur_depth = cur_depth+1
        search_list = new_search_list
    print("(cur_depth>0).sum()==grid**3",(grid_depth>=0).sum()==grid_size**3)
    return grid_depth
    
    
    
## 本次实现，不考虑噪声

def get_grid(pts, ptree, grid_size,sigma_set_minumum):
    
    sigma_set_minumum_value = sigma_set_minumum.max()
    # pts = np.loadtxt("%s%s.xyz"%(base_dir, shape))
    pts = normalize_input(pts)

    

    # normals = np.loadtxt("%s%s.normals"%(base_dir, shape))

    

    # grid_size = 80
    grid = np.zeros([grid_size, grid_size, grid_size],dtype=np.float64)
    N_shape = pts.shape[0]
    # print(N_shape)
    step = 2.0/grid_size
    # print(step)
    # for i in range(20):
    #     for j in range(20):
    #         for k in range(20):


    for i in range(N_shape):
        point = pts[i,:3]
        x_index = int((point[0] + 1.0) / step)
        y_index = int((point[1] + 1.0) / step)
        z_index = int((point[2] + 1.0) / step)
        # print(point[0],point[1],point[2])
        # print(x_index,y_index,z_index)
        # print(grid[x_index,y_index,z_index])
        grid[x_index,y_index,z_index] +=1

    
    grid_depth = get_grid_depth(grid)

        


        
    grid_new = copy.deepcopy(grid)
    directions = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                if grid[i,j,k]==0:
                    
                    for vi in [-1,0,1]:
                        for vj in [-1,0,1]:
                            for vk in [-1,0,1]:
                                # if abs(vi)==1 and abs(vj)==1 and abs(vk)==1:
                                #     continue
                                new_pos = i + vi, j + vj , k + vk
                                if noValid(new_pos,(grid_size, grid_size, grid_size)):
                                    continue
                                if grid[new_pos]>0:
                                    grid_new[i,j,k]=0.5
                                    break
                
                # 上面的循环未进入
                if grid_new[i,j,k]==0:
                    center_pts = (i+0.5) * step -1, (j+0.5) * step -1, (k+0.5) * step -1
                    d = ptree.query(center_pts, 1)
                    if sigma_set_minumum[d[1]] + step/2 * np.sqrt(2)> (d[0]):
                        grid_new[i,j,k]=0.5
                    # diff_dis = pts[d[1],:3] - pts[i:i+1,:3]
                    # normals = pts[d[1],3:]
                    # dis_pro = np.sum(diff_dis*normals)
                    # if d!=None:
                        
                        

    print("========!!!!!!!!!!!",(grid_new==0.5).sum())
    
    return grid_new,grid_depth

    # print((grid==0).sum()/(grid_size**3))


## 我们只是要算得最外面一圈对应的外部区域
## Note that, 这部分可能会不全，但是准确率是100%
def caculate_grid( grid, grid_depth, outdir, shape_name):
    size0, size1, size2 = grid.shape[0],grid.shape[1],grid.shape[2]
    # print(size0,size1,size2)
    visited_grid = np.zeros((size0, size1, size2), dtype=bool)
    search_list=[]
    num = 0
    # num_neg = 0
    # 将xy的两面添加进我们的search point
    x_dim_pos = 0
    x_dim_neg = size0-1
    # initialization
    for i in range(size1):
        for j in range(size2):
            if grid[x_dim_neg,i,j]==0 and not visited_grid[x_dim_neg,i,j]:
                search_list.append([x_dim_neg,i,j]) 
                num+=1
                visited_grid[x_dim_neg,i,j] = True
            if grid[x_dim_pos,i,j]==0 and not visited_grid[x_dim_pos,i,j]:
                search_list.append([x_dim_pos,i,j]) 
                num+=1
                visited_grid[x_dim_pos,i,j] = True
    y_dim_pos = 0
    y_dim_neg = size1-1
    for i in range(size0):
        for j in range(size2):
            if grid[i,y_dim_neg,j]==0 and not visited_grid[i,y_dim_neg,j]:
                search_list.append([i,y_dim_neg,j]) 
                num+=1
                visited_grid[i,y_dim_neg,j] = True
            if grid[i,y_dim_pos,j]==0 and not visited_grid[i,y_dim_pos,j]:
                search_list.append([i,y_dim_pos,j]) 
                num+=1
                visited_grid[i,y_dim_pos,j] = True
    
    z_dim_pos = 0
    z_dim_neg = size2-1
    for i in range(size0):
        for j in range(size1):
            if grid[i,j,z_dim_neg]==0 and not visited_grid[i,j,z_dim_neg]:
                search_list.append([i,j,z_dim_neg]) 
                num+=1
                visited_grid[i,j,z_dim_neg] = True
            if grid[i,j,z_dim_pos]==0 and not visited_grid[i,j,z_dim_pos]:
                search_list.append([i,j,z_dim_pos]) 
                num+=1
                visited_grid[i,j,z_dim_pos] = True


    print(np.sum(visited_grid==True))
    print(len(search_list))
    # outside_grid = np.bool8([size0, size1, size2])

    # loop
    step =0
    directions = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    for direction in directions:
        print(direction)
    # print("==============",len(search_list))
    while(step<len(search_list)):
        cur_pos = search_list[step]
        # print(cur_pos)
        # outside_grid[cur_pos[0],cur_pos[1],cur_pos[2]] = True
        for direction in directions:
            # new_pos = [0,0,0]
            new_pos = cur_pos[0] + direction[0], cur_pos[1] + direction[1] , cur_pos[2] + direction[2]
            # for i in range(3):
            #     new_pos[i] = cur_pos[i] + direction[i]
            # print(new_pos)
            # print(new_pos)
            if noValid(new_pos,(size0, size1, size2)):
                continue
            if visited_grid[new_pos[0],new_pos[1],new_pos[2]] or grid[new_pos[0],new_pos[1],new_pos[2]]>0:
                continue
            visited_grid[new_pos[0],new_pos[1],new_pos[2]] = True
            # print("==========")
            search_list.append([new_pos[0],new_pos[1],new_pos[2]])
            num+=1
        step+=1
    # print(size0)
    # print(size0**3)
    print(np.sum(visited_grid==True))
    print(np.sum(visited_grid==True)/(size0**3))
    print(len(search_list))
    
    
    
    
    
    grid_step = 2.0/size0
    center_pts = generate_boundary(size0,grid_step)
    for grid_index in search_list:
        center_pts.append([(grid_index[0]+0.5) * grid_step -1, (grid_index[1]+0.5) * grid_step -1, (grid_index[2]+0.5) * grid_step -1])
    
    pts = np.array(center_pts)
    np.savetxt("%s/%s_empty_%d_pts.txt"%(outdir, shape_name, size0),pts)
    print(pts.shape)


    other_pts = []
    for i in range(size0):
        for j in range(size1):
            for k in range(size2):
                if grid[i,j,k]<1 and not visited_grid[i,j,k]:
                    other_pts.append([(i+0.5) * grid_step -1, (j+0.5) * grid_step -1, (k+0.5) * grid_step -1, grid_depth[i,j,k]])

    other_grid = np.array(other_pts)
    np.savetxt("%s/%s_other_%d_pts.txt"%(outdir, shape_name, size0),other_grid)
    print(other_grid.shape)

def generate_boundary(size_grid, grid_step):
    
    '''
        generate the boundary outside the [-1,1]
    '''
    res_pts = []
    
    visited_grid = np.zeros((size_grid+2, size_grid+2, size_grid+2), dtype=bool)
    search_list = []
    num = 0
    # grid_step = 2.0/size_grid
    
    
    x_dim_pos = 0
    x_dim_neg = size_grid+1
    # initialization
    for i in range(size_grid+2):
        for j in range(size_grid+2):
            if not visited_grid[x_dim_neg,i,j]:
                search_list.append([x_dim_neg,i,j]) 
                num+=1
                visited_grid[x_dim_neg,i,j] = True
            if not visited_grid[x_dim_pos,i,j]:
                search_list.append([x_dim_pos,i,j]) 
                num+=1
                visited_grid[x_dim_pos,i,j] = True
    
    y_dim_pos = 0
    y_dim_neg = size_grid+1
    for i in range(size_grid+2):
        for j in range(size_grid+2):
            if not visited_grid[i,y_dim_neg,j]:
                search_list.append([i,y_dim_neg,j]) 
                num+=1
                visited_grid[i,y_dim_neg,j] = True
            if not visited_grid[i,y_dim_pos,j]:
                search_list.append([i,y_dim_pos,j]) 
                num+=1
                visited_grid[i,y_dim_pos,j] = True


    z_dim_pos = 0
    z_dim_neg = size_grid+1
    for i in range(size_grid+2):
        for j in range(size_grid+2):
            if not visited_grid[i,j,z_dim_neg]:
                search_list.append([i,j,z_dim_neg]) 
                num+=1
                visited_grid[i,j,z_dim_neg] = True
            if not visited_grid[i,j,z_dim_pos]:
                search_list.append([i,j,z_dim_pos]) 
                num+=1
                visited_grid[i,j,z_dim_pos] = True
    
    for grid_index in search_list:
        res_pts.append([(grid_index[0]+0.5-1) * grid_step -1, (grid_index[1]+0.5-1) * grid_step -1, (grid_index[2]+0.5-1) * grid_step -1])

    return res_pts
    
    
def noValid(new_pos,shape):
    for i in range(3):
        if new_pos[i] <0 or new_pos[i] >= shape[i]:
            return True
    return False 


def load_list(list_file_path):
    
    all_shape_list = []
    with open(list_file_path,"r") as f:
        all_shape_list = f.readlines()
    print(type(all_shape_list[0]))
    
    all_shape_list = [shape_name.strip() for shape_name in all_shape_list]
    print(all_shape_list)
    return all_shape_list

def normalize_save_abc(all_shape_list, all_data_input, all_data_normalized_save):
    for shape_name in all_shape_list:
        
        print(shape_name)
        pts = np.load("%s/%s.xyz.npy"%(all_data_input ,shape_name))
        pts = normalize_input(pts)

        pts_normal_info = getPCA_normal(pts,16)
        np.savetxt("%s/%s.txt"%(all_data_normalized_save ,shape_name),pts_normal_info)
        np.save("%s/%s.npy"%(all_data_normalized_save ,shape_name),pts_normal_info)

def generate_grid_res(all_shape_list, all_data_normalized_save, grid_outdir):
    for shape_name in all_shape_list:
        # file: E:/download/optim_data/thingi\120477.ply
    
        # print(file)
        # 120477
        # shape_name = file.split("\\")[1].split(".")[0]
        # shape_list.append(shape_name)
        # file = base_dir + shape_name + ".txt"
        # pts = np.loadtxt(file)
        # pts = normalize_input(pts)

        # pts_normal_info = getPCA_normal(pts,16)
        print(shape_name)
        pts = np.load("%s/%s.npy"%(all_data_normalized_save ,shape_name))[:,:3]
        # pts = normalize_input(pts)
        # pts = np.load("%s/%s.npy"%(all_data_normalized_save ,shape_name))[:,:3]

        # pcd = o3d.io.read_point_cloud(file)
        # pts = np.asarray(pcd.points)
        # normals = np.asarray(pcd.normals)
        # np.savetxt("case_pts.txt",np.concatenate([pts,normals],axis=1))
        sigma,sigma_set_minumum,ptree = calculate_density(pts)
        scale = int(2.0/(np.array(sigma).mean() * 1.5))
        size0 = scale//10 * 10
        if scale%10>=5:
            size0 +=   10
        print(scale)
        print(size0)
        # grid_new,grid_depth
        grid_res, grid_depth = get_grid(pts,ptree,size0,sigma_set_minumum)
        

        # size0= 80
        surface_pts = []
        grid_step = 2.0/size0
        for i in range(size0):
            for j in range(size0):
                for k in range(size0):
                    if grid_res[i,j,k]>=1:
                        surface_pts.append([(i+0.5) * grid_step -1, (j+0.5) * grid_step -1, (k+0.5) * grid_step -1])
        surface_pts = np.array(surface_pts)
        print(surface_pts.shape)
        np.savetxt("%s/%s_grid_%d_pts.txt"%(grid_outdir, shape_name, size0),surface_pts)
        caculate_grid(grid_res, grid_depth, grid_outdir, shape_name)

if __name__ == "__main__":
    
    # all_type = ["famous_dense", "famous_original", "famous_extra_noisy", "famous_noisefree", "famous_original", "famous_sparse"]
    all_type = ["abc_noisefree"]
    
    makedir("F:/local_processing_code/abc_all_small")
    for type_item in all_type:
        list_file_path = "C:/Users/runsongzhu/Downloads/erler-2020-p2s-abc/%s/testset.txt"%type_item
        grid_outdir = "F:/local_processing_code/output_res_1_%s_adaptive_more_grid11_density_boundary_depth"%type_item
        all_data_input = "C:/Users/runsongzhu/Downloads/erler-2020-p2s-abc/%s/04_pts/"%type_item
        all_data_normalized_save = "F:/local_processing_code/abc_all_small/%s"%type_item
        # all_files = glob.glob("%s/%s"%(base_dir, shape_post_name))
        all_shape_list = load_list(list_file_path)

        makedir(grid_outdir)
        makedir(all_data_normalized_save)
        print(all_shape_list)
        
        
    
        
        # normalize_save_abc(all_shape_list, all_data_input, all_data_normalized_save)
        generate_grid_res(all_shape_list, all_data_normalized_save, grid_outdir)
    
    
    # exit()
    


 


