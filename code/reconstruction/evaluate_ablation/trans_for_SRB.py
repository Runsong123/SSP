import trimesh
import numpy as np
import os

def my_mkdir(path):
  if not os.path.exists(path):
    os.mkdir(path)

#calculate the dict of mean,scale for each shape from raw input clouds
#shape_list = ["120477", "451676", "90276", "thingi_35269", "thingi_36371"]
shape_list = ["anchor", "daratech", "dc", "gargoyle", "lord_quas"]
input_dir = "/research/dept6/khhui/SSN_Fitting/raw_data/deep_geometric_prior_data/"
scale_mean_dict = dict()

scale_ouput_dir = "/research/d5/gds/rszhu22/SSN_Fitting_current/code_v8/reconstruction/evaluate_ablation/scaled_challenging_data/"
result_save_path = "/research/d5/gds/rszhu22/SSN_Fitting_current/code_v8/reconstruction/evaluate_ablation/result_signed_module"

for shape_item in shape_list:
  mesh = trimesh.load(input_dir+"%s.ply"%shape_item)
  mean_pts = np.mean(np.asarray(mesh.vertices),axis=0,keepdims=True)
  max_dis = np.linalg.norm((np.asarray(mesh.vertices) - mean_pts), ord=2, axis=1).max()
  scale_mean_dict[shape_item] = [mean_pts,max_dis]
  # mesh.vertices -= mean_pts
  # mesh.vertices /= max_dis
  
  # mesh.export(scale_ouput_dir+"%s/%s.ply"%("GT",shape_item))
my_mkdir(scale_ouput_dir)

# my_mkdir(scale_ouput_dir+"DeepMLS")
# #scale for each point clouds 
# base_dir = "/research/dept6/khhui/SSN_Fitting/code/code_v8/reconstruction/evaluate/challenging_data/DeepMLS/"

# for shape_item in shape_list:
#   mesh = trimesh.load(base_dir+"deep_geometric_prior_%s.obj"%shape_item)
#   mean_pts,max_dis = scale_mean_dict[shape_item]
#   mesh.vertices -= mean_pts
#   mesh.vertices /= max_dis
  
#   mesh.export(scale_ouput_dir+"%s/%s.ply"%("DeepMLS",shape_item))


# base_dir = "/research/dept6/khhui/SSN_Fitting/code/code_v8/reconstruction/evaluate/challenging_data/SAP/"
# my_mkdir(scale_ouput_dir+"SAP")
# for shape_item in shape_list:
#   mesh = trimesh.load(base_dir+"deep_geometric_prior_%s.ply"%shape_item)
#   mean_pts,max_dis = scale_mean_dict[shape_item]
#   mesh.vertices -= mean_pts
#   mesh.vertices /= max_dis
  
#   mesh.export(scale_ouput_dir+"%s/%s.ply"%("SAP",shape_item))



# base_dir = "/research/dept6/khhui/SSN_Fitting/code/code_v8/reconstruction/evaluate/challenging_data/NP/"
# my_mkdir(scale_ouput_dir+"NP")
# # shape_list = ["120477", "451676", "90276", "thingi_35269", "thingi_36371"]
# for shape_item in shape_list:
#   mesh = trimesh.load(base_dir+"occn_%s_0.005.off"%shape_item)
#   #mean_pts,max_dis = scale_mean_dict[shape_item]
#   #mesh.vertices -= mean_pts
#   mesh.vertices *= 1.1
  
#   mesh.export(scale_ouput_dir+"%s/%s.ply"%("NP",shape_item))


base_dir = result_save_path + "/derivative_ablation/"
my_mkdir(scale_ouput_dir+"derivative_ablation")
# shape_list = ["120477", "451676", "90276", "thingi_35269", "thingi_36371"]
for shape_item in shape_list:
  mesh = trimesh.load(base_dir+"igr_20000_%s.ply"%shape_item)
  mesh.vertices *= 1.1
  
  mesh.export(scale_ouput_dir+"%s/%s.ply"%("derivative_ablation",shape_item))


base_dir = result_save_path + "/signed_ablation/"
# base_dir = "/research/dept6/khhui/SSN_Fitting/code/code_v8/reconstruction/evaluate/challenging_data/signed_ablation/"
my_mkdir(scale_ouput_dir+"signed_ablation")
# shape_list = ["120477", "451676", "90276", "thingi_35269", "thingi_36371"]
for shape_item in shape_list:
  mesh = trimesh.load(base_dir+"igr_20000_%s.ply"%shape_item)
  mesh.vertices *= 1.1
  
  mesh.export(scale_ouput_dir+"%s/%s.ply"%("signed_ablation",shape_item))


base_dir = result_save_path + "/unsigned_ablation/"
# base_dir = "/research/dept6/khhui/SSN_Fitting/code/code_v8/reconstruction/evaluate/challenging_data/signed_ablation/"
my_mkdir(scale_ouput_dir+"unsigned_ablation")
# shape_list = ["120477", "451676", "90276", "thingi_35269", "thingi_36371"]
for shape_item in shape_list:
  mesh = trimesh.load(base_dir+"igr_20000_%s.ply"%shape_item)
  mesh.vertices *= 1.1
  
  mesh.export(scale_ouput_dir+"%s/%s.ply"%("unsigned_ablation",shape_item))


# base_dir = "/research/dept6/khhui/SSN_Fitting/code/code_v8/reconstruction/evaluate/challenging_data/compare_traditional_sample/"
# my_mkdir(scale_ouput_dir+"compare_traditional_sample")
# # shape_list = ["120477", "451676", "90276", "thingi_35269", "thingi_36371"]
# for shape_item in shape_list:
#   mesh = trimesh.load(base_dir+"igr_20000_%s.ply"%shape_item)
#   mesh.vertices *= 1.1
  
#   mesh.export(scale_ouput_dir+"%s/%s.ply"%("compare_traditional_sample",shape_item))

  
  


  