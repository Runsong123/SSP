import logging
import numpy as np
import trimesh
from pykdtree.kdtree import KDTree
from plyfile import PlyData
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

EMPTY_PCL_DICT = {
    "completeness": np.sqrt(3),
    "accuracy": np.sqrt(3),
    "completeness2": 3,
    "accuracy2": 3,
    "chamfer": 6,
}

EMPTY_PCL_DICT_NORMALS = {
    "normals completeness": -1.0,
    "normals accuracy": -1.0,
    "normals": -1.0,
}

logger = logging.getLogger(__name__)


class MeshEvaluator(object):
    """ Mesh evaluation class.
    It handles the mesh evaluation process.
    Args:
        n_points (int): number of points to be used for evaluation
    """
    def __init__(self, n_points=100000):
        self.n_points = n_points

    def eval_mesh(self,
                  obj_id,
                  mesh,
                  pointcloud_tgt,
                  normals_tgt,
                  save_dir,
                  thresholds=np.linspace(1.0 / 1000, 1, 1000),
                  center=None,
                  scale=None):
        """ Evaluates a mesh.
        Args:
            mesh (trimesh): mesh which should be evaluated
            pointcloud_tgt (numpy array): target point cloud
            normals_tgt (numpy array): target normals
            thresholds (numpy arry): for F-Score
        """
        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            pointcloud, idx = mesh.sample(self.n_points, return_index=True)

            pointcloud = pointcloud.astype(np.float32)
            normals = mesh.face_normals[idx]
        else:
            pointcloud = np.empty((0, 3))
            normals = np.empty((0, 3))

        out_dict = self.eval_pointcloud(obj_id,
                                        pointcloud,
                                        pointcloud_tgt,
                                        normals,
                                        normals_tgt,
                                        save_dir,
                                        thresholds=thresholds,
                                        center=center,
                                        scale=center)

        return out_dict

    def eval_pointcloud(self,
                        obj_id,
                        pointcloud,
                        pointcloud_tgt,
                        normals=None,
                        normals_tgt=None,
                        save_dir=None,
                        thresholds=np.linspace(1.0 / 1000, 1, 1000),
                        center=None,
                        scale=None):
        """ Evaluates a point cloud.
        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
            thresholds (numpy array): threshold values for the F-score calculation
        """
        # Return maximum losses if pointcloud is empty
        if pointcloud.shape[0] == 0:
            logger.warn("Empty pointcloud / mesh detected!")
            out_dict = EMPTY_PCL_DICT.copy()
            if normals is not None and normals_tgt is not None:
                out_dict.update(EMPTY_PCL_DICT_NORMALS)
            return out_dict

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(pointcloud_tgt, normals_tgt, pointcloud, normals)
        recall = get_threshold_percentage(completeness, thresholds)
        completeness2 = completeness**2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(pointcloud, normals, pointcloud_tgt, normals_tgt)
        precision = get_threshold_percentage(accuracy, thresholds)
        accuracy2 = accuracy**2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        normals_correctness = 0.5 * completeness_normals + 0.5 * accuracy_normals
        chamferL1 = 0.5 * (completeness + accuracy)

        # F-Score
        F = [2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-10) for i in range(len(precision))]

        out_dict = {
            "id": obj_id,
            "center": center,
            "scale": scale,
            "pp": pointcloud,
            "tp": pointcloud_tgt,
            "pn": normals,
            "tn": normals_tgt,
            "completeness": completeness,
            "accuracy": accuracy,
            "normals completeness": completeness_normals,
            "normals accuracy": accuracy_normals,
            "normals": normals_correctness,
            "completeness2": completeness2,
            "accuracy2": accuracy2,
            "chamfer-L2": chamferL2,
            "chamfer-L1": chamferL1,
            "F": F,
            "f-score": F[9],  # threshold = 1.0%
            "f-score-15": F[14],  # threshold = 1.5%
            "f-score-20": F[19],  # threshold = 2.0%
        }
        np.save("./" + save_dir + "/" + obj_id + ".npy", out_dict)

        return out_dict


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    """ Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    """
    kdtree = KDTree(points_tgt)
    print(points_tgt.shape)
    print(points_src.shape)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array([np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def get_threshold_percentage(dist, thresholds):
    """ Evaluates a point cloud.
    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    """
    in_threshold = [(dist <= t).mean() for t in thresholds]
    return in_threshold


def check_missing_part(exp_name, shape_list, epoch):

    missing_part = []
    for shape_item in shape_list:
        exp_base_dir = "/apdcephfs/share_1467498/datasets/runsongzhu/IGR/%s/exps/" % exp_name

        timestamps = os.listdir(os.path.join(exp_base_dir, shape_item))
        timestamp = sorted(timestamps)[-1]
        print(timestamp)

        file_data = "%s/%s/%s/plots/igr_%s_%s.ply" % (exp_base_dir, shape_item, timestamp, epoch, shape_item)

        if not os.path.exists(file_data):
            missing_part.append(shape_item)

    return missing_part


def myMkdir(input_path):
    if not os.path.exists(input_path):
        os.mkdir(input_path)


def generate_dir():
    data2SaveMeshDir = dict()
    data2SaveMeshDir["abc_noisefree"] = "abc_noisefree"
    all_datasets = ["low_noise", "med_noise", "high_noise"]
    for data_type in all_datasets:
        data2SaveMeshDir[data_type] = "%s/%s" % ("noisy", data_type)

    all_datasets = ["vardensity_gradient", "vardensity_striped"]

    for data_type in all_datasets:
        data2SaveMeshDir[data_type] = "%s/%s" % ("density", data_type)

    return data2SaveMeshDir


if __name__ == "__main__":

    import glob
    # shape_list = ["120477", "451676", "90276", "thingi_35269", "thingi_36371"]
    # dataset_name = "thingi"
    # exp_name = "things_full_model_by_PCA_normal_SAL_v9"
    # exp_name_2 = "things_full_model_by_PCA_normal_SAL_v10"
    # exp_name_all = ["things_full_model_by_PCA_normal_SAL_v9", "things_full_model_by_PCA_normal_SAL_v10"]
    # exp_name = "things_full_model_by_PCA_normal_geometry_prior_data_80_SAL_1"
    # data_type = ["abc_noisefree", "density_variation", "noisy_data"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str, help="all or hard case")
    args = parser.parse_args()
    if not args.part == "all" and not args.part == "DiGS_hard":
        print('set right datatype')
        exit()

    if args.part == "all":
        file_name = "/research/d5/gds/rszhu22/SSN_Fitting_current/data/raw_data/optim_data/abc_noisefree/testset.txt"
        base_dir = "/research/d5/gds/rszhu22/SSN_Fitting_current/code/reconstruction/evaluate/abc_CAD_data/"
    elif args.part == "DiGS_hard":
        file_name = "/research/d5/gds/rszhu22/DiGS/surface_reconstruction/hard_case.txt"
        base_dir = "/research/d5/gds/rszhu22/SSN_Fitting_current/code/reconstruction/evaluate/abc_CAD_data_hard/"

    with open(file_name, 'r') as f:
        shape_list = f.readlines()
    shape_list = [file_name.strip() for file_name in shape_list]
    shape_list = shape_list

    data_type = ["abc_noisefree"]
    # data_type = ["density_variation"]
    epoch = 20000  # DEFAULT: 20,000
    # methods_name = ["SSN_fitting_ablantion_v1","ablation_balance", "ablation_balance_v2"]
    # methods_name = [ "SAP", "NeuralPull", "IGR", "PSR", "SAL","SALD", "DMLS", "run_SSN_Fitting_code_v8_run_SSN_Fitting_thinigi_new_1_v1"]
    # methods_name = ["SAP", "SALD", "run_SSN_Fitting_code_v8_run_SSN_Fitting_thinigi_new_1_v1"]
    # methods_name = ["SAP", "SALD", "run_SSN_Fitting_code_v8_run_SSN_Fitting_thinigi_new_1_v1"]
    # base_dir = "/research/d5/gds/rszhu22/SSN_Fitting_current/code_v8/reconstruction/evaluate/scaled_challenging_data/"
    # base_dir = "/research/d5/gds/rszhu22/SSN_Fitting_current/code/reconstruction/evaluate/abc_CAD_data/"
    myMkdir(base_dir)
    all_datatype_dir = generate_dir()
    # methods_name = ["SSNFitting_CVPR", "SAP", "DMLS"]
    # methods_name = [
    #     "SSNFitting_CVPR", "SSNFitting_CVPR_new_clip", "SSNFitting_CVPR_3000", "SSNFitting_CVPR_8000", "SSNFitting_CVPR_8000_updated"
    # ]
    # methods_name = ["DiGS", "DiGS_signed_supervision"]
    # methods_name = ["DiGS_official_epoch", "SAL"]
    # methods_name = ["DiGS_official_epoch", "DiGS_official_epoch_signed_supervision", "SSNFitting_CVPR_10000_EPOCH"]
    # methods_name = ["DiGS_official_epoch", "SSNFitting_CVPR_10000_EPOCH", "SAL", "SALD", "SAP", "PSR", "POCO", "DiGS_w_minimal_surface"]
    # methods_name = ["DiGS_official_epoch", "POCO", "DiGS_w_minimal_surface"]
    methods_name = [
        "DiGS_official_epoch", "DiGS_w_minimal_surface", "DiGS_w_minimal_v14", "DiGS_w_minimal_v22"
        # , "DiGS_w_minimal_v7_hard", "DiGS_w_minimal_v6_hard", "DiGS_w_minimal_v8_hard", "DiGS_w_minimal_v9_hard",
        # "DiGS_w_minimal_v10_hard"
    ]

    # print(methods_name[-1])
    # methods_name = ["SSN_fitting_ablantion_v8_all_v16","SSN_fitting_ablantion_v8_all_v8","SSN_fitting_ablantion_v8_all_v1", "SSN_fitting_ablantion_v8_all", "run_SSN_Fitting_v8_all_focal", "run_SSN_Fitting_v8_all_focal_v1"]
    # methods_name = ["ablation_balance_11", "ablation_balance_v2", "ablation_importance", "ablation_signed_module"]
    gt_cache = {}
    n_points = 40000
    evaluator = MeshEvaluator(n_points=n_points)
    device = "cpu"

    print("good!! and  check")
    metric_save_dir = "metric_main_compares_abc%s" % args.part
    if not os.path.exists(metric_save_dir):
        os.mkdir(metric_save_dir)

    # epoch = 20000
    # for type_id, data_type in enumerate(all_types):

    # for idx, type_item in enumerate(data_type):
    compare_num = len(methods_name)
    f_10 = [[] for i in range(compare_num)]
    f_15 = [[] for i in range(compare_num)]
    f_20 = [[] for i in range(compare_num)]
    F = [[] for i in range(compare_num)]
    # print(F)
    CD_1 = [[] for i in range(compare_num)]
    CD_2 = [[] for i in range(compare_num)]
    NC = [[] for i in range(compare_num)]
    object_id = [[] for i in range(compare_num)]

    for type_item in ["abc_noisefree"]:

        for method_idx, method_item in enumerate(methods_name):
            print(method_item)

            save_dir = "abc_%s_%s_%dp_epoch_final_compare_all" % (method_item, type_item, n_points)
            if not os.path.exists(save_dir):
                print(save_dir)
                os.mkdir(save_dir)

            for shape_item in shape_list:

                shape_name = shape_item
                if type_item[:7] == "density":
                    original_shape_name = shape_name.split("_d")[0]
                elif type_item[:5] == "noise":
                    original_shape_name = shape_name.split("_n")[0]
                else:
                    original_shape_name = shape_name

                gt_shape_path = "%s/GT/%s.ply" % (
                    base_dir,
                    shape_item,
                )

                npy_path = "%s/%s.npy" % (save_dir, original_shape_name)

                if os.path.exists(npy_path):

                    data = np.load(npy_path, allow_pickle=True)

                    ######### fixbug!!!!!!!!!!!!
                    gt_pointcloud = data.item()["tp"]
                    gt_normals = data.item()["tn"]
                    scale = data.item()["scale"]
                    center = data.item()["center"]

                    f_10[method_idx].append(data.item()["f-score"])
                    f_15[method_idx].append(data.item()["f-score-15"])
                    f_20[method_idx].append(data.item()["f-score-20"])
                    NC[method_idx].append(data.item()["normals"])
                    CD_1[method_idx].append(data.item()["chamfer-L1"])
                    CD_2[method_idx].append(data.item()["chamfer-L2"])
                    print(data.item()["f-score"])
                else:
                    # original_name = shape_item.split("_dd")[0]
                    if gt_shape_path in gt_cache.keys():
                        gt_pointcloud, gt_normals, center, scale = gt_cache[gt_shape_path]
                        print("use gt")
                    else:
                        gt_mesh = trimesh.load_mesh(gt_shape_path)

                        if isinstance(gt_mesh, trimesh.Trimesh):
                            gt_pointcloud, idx_pts = gt_mesh.sample(n_points, return_index=True)
                            gt_pointcloud = gt_pointcloud.astype(np.float32)
                            gt_normals = gt_mesh.face_normals[idx_pts]
                        elif isinstance(gt_mesh, trimesh.PointCloud):
                            print(";;;;")
                            # gt_pointcloud = gt_mesh.vertices.astype(np.float32)
                            # gt_normals = None
                            plydata = PlyData.read(gt_shape_path)
                            gt_pointcloud = np.stack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]], axis=1)
                            gt_normals = np.stack([plydata["vertex"]["nx"], plydata["vertex"]["ny"], plydata["vertex"]["nz"]], axis=1)
                        else:
                            raise RuntimeError("Unknown data type!")

                        N = gt_pointcloud.shape[0]
                        center = gt_pointcloud.mean(0)
                        scale = np.abs(gt_pointcloud - center).max()
                        # print(center, scale)

                        gt_pointcloud -= center
                        gt_pointcloud /= scale

                        gt_cache.update({gt_shape_path: (gt_pointcloud, gt_normals, center, scale)})

                    target_pts = gt_pointcloud
                    target_normals = gt_normals

                    if True:

                        # exp_base_dir = "/apdcephfs/share_1467498/datasets/runsongzhu/IGR/%s/exps/" % exp_name_list[method_idx]
                        # timestamps = os.listdir(os.path.join(exp_base_dir, shape_item))
                        # timestamp = sorted(timestamps)[-1]
                        file_data = "%s/%s/%s.ply" % (
                            base_dir,
                            method_item,
                            shape_item,
                        )

                        if not os.path.exists(file_data):
                            print(file_data)
                            continue
                        mesh = trimesh.load(file_data, process=False)
                        print("check this!!!!", np.linalg.norm(mesh.vertices, ord=2, axis=1).max())

                    mesh.vertices -= center
                    mesh.vertices /= scale
                    # scale = 1.
                    # print(np.abs(gt_pointcloud).max())
                    thresholds = np.linspace(1.0 / 1000, 1, 1000)
                    eval_dict_mesh = evaluator.eval_mesh(shape_item,
                                                         mesh,
                                                         target_pts,
                                                         target_normals,
                                                         save_dir,
                                                         thresholds=thresholds,
                                                         center=center,
                                                         scale=center)
                    # print(eval_dict_mesh)
                    # print(eval_dict_mesh["f-score"])
                    f_10[method_idx].append(eval_dict_mesh["f-score"])
                    f_15[method_idx].append(eval_dict_mesh["f-score-15"])
                    f_20[method_idx].append(eval_dict_mesh["f-score-20"])
                    NC[method_idx].append(eval_dict_mesh["normals"])
                    CD_1[method_idx].append(eval_dict_mesh["chamfer-L1"])
                    CD_2[method_idx].append(eval_dict_mesh["chamfer-L2"])

                # acc_metric[idx].append(eval_dict_mesh["accuracy"])
                # complete_metric[idx].append(eval_dict_mesh["completeness"])

    save_data = {}
    save_data["methods"] = methods_name
    save_data["score_10"] = [round(np.array(f_10_item).mean(), 3) for f_10_item in f_10]  #np.mean(np.array(f_10))
    # save_data["score_15"] = [round(np.array(f_15_item).mean(), 3) for f_15_item in f_15]  #np.mean(np.array(f_15[idx]))
    # save_data["score_20"] = [round(np.array(f_20_item).mean(), 3) for f_20_item in f_20]  #np.mean(np.array(f_20))
    save_data["CD1"] = [round(np.array(CD_1_item).mean() * 100.0, 3) for CD_1_item in CD_1]  #np.mean(np.array(CD_1))
    # save_data["CD1"] = [round(np.array(CD_2_item).mean() * 100.0, 3) for CD_2_item in CD_2]  #np.mean(np.array(CD_1))
    # save_data["CD2"] = [ np.array(CD_2_item).mean() for CD_2_item in CD_2]# np.mean(np.array(CD_2))
    save_data["NC"] = [round(np.array(NC_item).mean(), 3) for NC_item in NC]  # np.mean(np.array(NC))

    top_list = list(np.argsort(np.array(CD_1[-1]) - np.array(CD_1[0])))
    # print(CD_1[-1][top_list[0]])
    # print(CD_2[0][top_list[1]])
    # # print(top_list)
    # # print(object_id)
    # top_list = [[object_id[idx][name_idx] for name_idx in top_list_item] for idx, top_list_item in enumerate(top_list)]
    # print(top_list)
    # import shutil
    print("---------------------")
    shape_check_list = []
    for item in top_list[::-1]:
        print(CD_1[-1][item])
        shape_check_list.append(shape_list[item])
    with open("failure_case_worsen.txt", "w") as f:
        for shape_item in shape_check_list:
            f.write(shape_item + "\n")

    # if args.part == "all":
    #     top_list = (list(np.argsort(np.array(CD_1[-1]))))[-10:]
    #     shape_check_list = []
    #     for item in top_list:
    #         shape_check_list.append(shape_list[item])
    #     import shutil
    #     check_shape_dir = "/research/d5/gds/rszhu22/SSN_Fitting_current/code/reconstruction/evaluate/abc_CAD_data/DiGS/hard/"
    #     myMkdir(check_shape_dir)
    #     for shape_item in shape_check_list:
    #         source_file_data = "%s/%s/%s.ply" % (
    #             base_dir,
    #             "DiGS",
    #             shape_item,
    #         )
    #         shutil.copy(source_file_data, check_shape_dir + "%s.ply" % shape_item)

    #     with open("hard_case.txt", "w") as f:
    #         for shape_item in shape_check_list:
    #             f.write(shape_item + "\n")

    df = pd.DataFrame(save_data)
    df.to_csv("%s/abc_compare_all_digs.csv" % (metric_save_dir))
