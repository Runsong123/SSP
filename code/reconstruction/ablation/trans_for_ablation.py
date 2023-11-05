import trimesh
import numpy as np
import os


def my_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


#calculate the dict of mean,scale for each shape from raw input clouds
shape_list = ["120477", "451676", "90276", "thingi_35269", "thingi_36371"]
input_dir = "/research/dept6/khhui/shape_as_points/data/thingi/"
scale_mean_dict = dict()
# for shape_item in shape_list:
#   mesh = trimesh.load(input_dir+"%s.ply"%shape_item)
#   mean_pts = np.mean(np.asarray(mesh.vertices),axis=0,keepdims=True)
#   max_dis = np.linalg.norm((np.asarray(mesh.vertices) - mean_pts), ord=2, axis=1).max()
#   scale_mean_dict[shape_item] = [mean_pts,max_dis]

# scale_ouput_dir = "/research/dept6/khhui/SSN_Fitting/code/code_v8/reconstruction/evaluate/scaled_challenging_data/"
scale_ouput_dir = "/research/d5/gds/rszhu22/SSN_Fitting_current/code/reconstruction/evaluate/scaled_challenging_data/"
# scale_ouput_dir = "/research/d5/gds/rszhu22/SSN_Fitting_current/code_v8/evaluate/scaled_challenging_data/"
my_mkdir(scale_ouput_dir)
# my_mkdir(scale_ouput_dir + "input_pts")
# for shape_item in shape_list:
#     mesh = trimesh.load(input_dir + "%s.ply" % shape_item)
#     mean_pts = np.mean(np.asarray(mesh.vertices), axis=0, keepdims=True)
#     max_dis = np.linalg.norm((np.asarray(mesh.vertices) - mean_pts), ord=2, axis=1).max()
#     # scale_mean_dict[shape_item] = [mean_pts,max_dis]
#     mesh.vertices -= mean_pts
#     mesh.vertices /= max_dis

#     mesh.export(scale_ouput_dir + "%s/%s.ply" % ("input_pts", shape_item))

# exit()

# my_mkdir(scale_ouput_dir+"DeepMLS")
# #scale for each point clouds
# base_dir = "/research/dept6/khhui/SSN_Fitting/code/code_v8/reconstruction/evaluate/challenging_data/DeepMLS/"

# for shape_item in shape_list:
#   mesh = trimesh.load(base_dir+"%s.obj"%shape_item)
#   mean_pts,max_dis = scale_mean_dict[shape_item]
#   mesh.vertices -= mean_pts
#   mesh.vertices /= max_dis

#   mesh.export(scale_ouput_dir+"%s/%s.ply"%("DeepMLS",shape_item))

# base_dir = "/research/dept6/khhui/SSN_Fitting/code/code_v8/reconstruction/evaluate/challenging_data/SAP/"
# my_mkdir(scale_ouput_dir+"SAP")
# for shape_item in shape_list:
#   mesh = trimesh.load(base_dir+"%s.ply"%shape_item)
#   mean_pts,max_dis = scale_mean_dict[shape_item]
#   mesh.vertices -= mean_pts
#   mesh.vertices /= max_dis

#   mesh.export(scale_ouput_dir+"%s/%s.ply"%("SAP",shape_item))

# base_dir = "/research/dept6/khhui/SSN_Fitting/code/code_v8/reconstruction/evaluate/challenging_data/NP/"
# my_mkdir(scale_ouput_dir+"NP")
# shape_list = ["120477", "451676", "90276", "thingi_35269", "thingi_36371"]
# for shape_item in shape_list:
#   mesh = trimesh.load(base_dir+"occn_%s_0.005.off"%shape_item)
#   #mean_pts,max_dis = scale_mean_dict[shape_item]
#   #mesh.vertices -= mean_pts
#   mesh.vertices *= 1.1

#   mesh.export(scale_ouput_dir+"%s/%s.ply"%("NP",shape_item))

# base_dir = "/research/dept6/khhui/SSN_Fitting/code/code_v8/reconstruction/evaluate/challenging_data/Our/"
# my_mkdir(scale_ouput_dir + "Our_8000")
# shape_list = ["120477", "451676", "90276", "thingi_35269", "thingi_36371"]
# for shape_item in shape_list:
#     mesh = trimesh.load(base_dir + "igr_20000_%s.ply" % shape_item)
#     mesh.vertices *= 1.1

#     mesh.export(scale_ouput_dir + "%s/%s.ply" % ("Our", shape_item))

# my_mkdir(scale_ouput_dir+"GT")
# shape_list = ["120477", "451676", "90276", "thingi_35269", "thingi_36371"]
# for shape_item in shape_list:
#   mesh = trimesh.load(input_dir+"%s.obj"%shape_item)
#   mean_pts,max_dis = scale_mean_dict[shape_item]
#   mesh.vertices -= mean_pts
#   mesh.vertices /= max_dis

#   mesh.export(scale_ouput_dir+"%s/%s.ply"%("GT",shape_item))

# base_dir = "/research/dept6/khhui/SSN_Fitting/code/code_v8/reconstruction/evaluate/challenging_data/derivative_ablation/"
# my_mkdir(scale_ouput_dir+"derivative_ablation")
# shape_list = ["120477", "451676", "90276", "thingi_35269", "thingi_36371"]
# for shape_item in shape_list:
#   mesh = trimesh.load(base_dir+"igr_20000_%s.ply"%shape_item)
#   mesh.vertices *= 1.1

#   mesh.export(scale_ouput_dir+"%s/%s.ply"%("derivative_ablation",shape_item))

# base_dir = "/research/dept6/khhui/SSN_Fitting/code/code_v8/reconstruction/evaluate/challenging_data/signed_ablation/"
# my_mkdir(scale_ouput_dir+"signed_ablation")
# shape_list = ["120477", "451676", "90276", "thingi_35269", "thingi_36371"]
# for shape_item in shape_list:
#   mesh = trimesh.load(base_dir+"igr_20000_%s.ply"%shape_item)
#   mesh.vertices *= 1.1

#   mesh.export(scale_ouput_dir+"%s/%s.ply"%("signed_ablation",shape_item))

# base_dir = "/research/dept6/khhui/SSN_Fitting/code/code_v8/reconstruction/evaluate/challenging_data/compare_traditional_sample/"
# my_mkdir(scale_ouput_dir+"compare_traditional_sample")
# shape_list = ["120477", "451676", "90276", "thingi_35269", "thingi_36371"]
# for shape_item in shape_list:
#   mesh = trimesh.load(base_dir+"igr_20000_%s.ply"%shape_item)
#   mesh.vertices *= 1.1

#   mesh.export(scale_ouput_dir+"%s/%s.ply"%("compare_traditional_sample",shape_item))
# /research/d5/gds/rszhu22/SSN_Fitting_current/code_v8/empty_uniform_v5_precalc_v2/
# base_dir = "/research/d5/gds/rszhu22/SSN_Fitting_current/code_v8/reconstruction/famous/"
# my_mkdir(scale_ouput_dir + "SSNFitting_CVPR")
# shape_list = ["120477", "451676", "90276", "thingi_35269", "thingi_36371"]
# for shape_item in shape_list:
#     mesh = trimesh.load(base_dir + "igr_8000_%s.ply" % shape_item)
#     mesh.vertices *= 1.1

#     mesh.export(scale_ouput_dir + "%s/%s.ply" % ("SSNFitting_CVPR", shape_item))

# /research/d5/gds/rszhu22/SSN_Fitting_current/code_v8/empty_uniform_v5_precalc_v2/
# base_dir = "/research/d5/gds/rszhu22/SSN_Fitting_current/code_v8/reconstruction/challenging/"
# method_name = "SSNFitting_CVPR_10000_EPOCH"
# my_mkdir(scale_ouput_dir + method_name)
# EPOCH = 10000
# shape_list = ["120477", "451676", "90276", "thingi_35269", "thingi_36371"]
# for shape_item in shape_list:
#     mesh = trimesh.load(base_dir + "igr_%d_%s.ply" % (EPOCH, shape_item))
#     mesh.vertices *= 1.1

#     mesh.export(scale_ouput_dir + "%s/%s.ply" % (method_name, shape_item))

# # ##################### DiSG
# base_dir = "/research/d5/gds/rszhu22/DiGS/surface_reconstruction/log/surface_reconstruction/DiGS_surf_recon_experiment/result_meshes_10000/"
# method_name = "DiGS_official_epoch"
# my_mkdir(os.path.join(scale_ouput_dir, method_name))
# # shape_list = ["120477", "451676", "90276", "thingi_35269", "thingi_36371"]
# missing_shape_list = []
# for shape_item in shape_list:
#     if not os.path.exists(base_dir + "%s.ply" % shape_item):
#         missing_shape_list.append(shape_item)
#         continue
#     mesh = trimesh.load(base_dir + "%s.ply" % shape_item)
#     mesh.vertices *= 1.1
#     mesh.export(scale_ouput_dir + "%s/%s.ply" % (method_name, shape_item))
# with open("DiSG_missing.txt", "w") as f:
#     for missing_item in missing_shape_list:
#         f.write("%s\n" % missing_item)

# base_dir = "/research/d5/gds/rszhu22/DiGS/surface_reconstruction/log/surface_reconstruction/DiGS_surf_recon_experiment/result_meshes/"
# my_mkdir(scale_ouput_dir + "DiGS")

# shape_list = ["120477", "451676", "90276", "thingi_35269", "thingi_36371"]
# for shape_item in shape_list:
#     mesh = trimesh.load(base_dir + "%s.ply" % shape_item)
#     mesh.vertices *= 1.1

#     mesh.export(scale_ouput_dir + "%s/%s.ply" % ("DiGS", shape_item))

# EPOCH = 10000

# Method = "ABLATION_TS"
# my_mkdir(scale_ouput_dir + Method)
# base_dir = "/research/d5/gds/rszhu22/SSN_Fitting_current/code_v8/exps/%s/exps" % Method
# for shape_item in shape_list:

#     timestamps = os.listdir(os.path.join(base_dir, shape_item))
#     timestamp = os.path.join(base_dir, shape_item, sorted(timestamps)[-1])
#     mesh = trimesh.load(timestamp + "/plots/igr_%d_%s.ply" % (EPOCH, shape_item))
#     mesh.vertices *= 1.1
#     mesh.export(scale_ouput_dir + "%s/%s.ply" % (Method, shape_item))

# Method = "ABLATION_TS_16384"
# my_mkdir(scale_ouput_dir + Method)
# base_dir = "/research/d5/gds/rszhu22/SSN_Fitting_current/code_v8/exps/%s/exps" % Method
# for shape_item in shape_list:

#     timestamps = os.listdir(os.path.join(base_dir, shape_item))
#     timestamp = os.path.join(base_dir, shape_item, sorted(timestamps)[-1])
#     mesh = trimesh.load(timestamp + "/plots/igr_%d_%s.ply" % (EPOCH, shape_item))
#     mesh.vertices *= 1.1
#     mesh.export(scale_ouput_dir + "%s/%s.ply" % (Method, shape_item))

# Method = "ABLATION_SIGNED"
# my_mkdir(scale_ouput_dir + Method)
# base_dir = "/research/d5/gds/rszhu22/SSN_Fitting_current/code_v8/exps/%s/exps" % Method
# for shape_item in shape_list:

#     timestamps = os.listdir(os.path.join(base_dir, shape_item))
#     timestamp = os.path.join(base_dir, shape_item, sorted(timestamps)[-1])
#     mesh = trimesh.load(timestamp + "/plots/igr_%d_%s.ply" % (EPOCH, shape_item))
#     mesh.vertices *= 1.1
#     mesh.export(scale_ouput_dir + "%s/%s.ply" % (Method, shape_item))

# Method = "ABLATION_DS"
# my_mkdir(scale_ouput_dir + Method)
# base_dir = "/research/d5/gds/rszhu22/SSN_Fitting_current/code_v8/exps/%s/exps" % Method
# for shape_item in shape_list:

#     timestamps = os.listdir(os.path.join(base_dir, shape_item))
#     timestamp = os.path.join(base_dir, shape_item, sorted(timestamps)[-1])
#     mesh = trimesh.load(timestamp + "/plots/igr_%d_%s.ply" % (EPOCH, shape_item))
#     mesh.vertices *= 1.1
#     mesh.export(scale_ouput_dir + "%s/%s.ply" % (Method, shape_item))

# Method = "ABLATION_PE"
# my_mkdir(scale_ouput_dir + Method)
# base_dir = "/research/d5/gds/rszhu22/SSN_Fitting_current/code_v8/exps/%s/exps" % Method
# for shape_item in shape_list:

#     timestamps = os.listdir(os.path.join(base_dir, shape_item))
#     timestamp = os.path.join(base_dir, shape_item, sorted(timestamps)[-1])
#     mesh = trimesh.load(timestamp + "/plots/igr_%d_%s.ply" % (EPOCH, shape_item))
#     mesh.vertices *= 1.1
#     mesh.export(scale_ouput_dir + "%s/%s.ply" % (Method, shape_item))


# EPOCH = 20000
# Method = "IGR_calculated_oriented"
# my_mkdir(scale_ouput_dir + Method)
# base_dir = "/research/d5/gds/rszhu22/SSN_Fitting_current/code_v8/exps/%s/exps" % Method
# for shape_item in shape_list:

#     timestamps = os.listdir(os.path.join(base_dir, shape_item))
#     timestamp = os.path.join(base_dir, shape_item, sorted(timestamps)[-1])
#     mesh = trimesh.load(timestamp + "/plots/igr_%d_%s.ply" % (EPOCH, shape_item))
#     mesh.vertices *= 1.1
#     mesh.export(scale_ouput_dir + "%s/%s.ply" % (Method, shape_item))





