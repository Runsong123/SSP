import struct
import base64
import tripy
import gltflib
import trimesh

def ply_to_glb(ply_filename, glb_filename):
    m = trimesh.load(ply_filename,  split_object=True, group_material=False, process=False)
    m.export('output.glb')


if __name__ == "__main__":
    ply_to_glb("/Users/zhuzhuxia/SSP_web/mesh/Our/90276.ply", "output.glb")
