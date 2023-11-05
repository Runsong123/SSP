# shape_list=("120477" "451676" "90276" "thingi_35269" "thingi_36371")
class_list=("loudspeaker" "rifle" "sofa")
for class in  ${class_list[@]};
do
# shape_list=ls /home/yzhang4/DATA/ShapeNetCore.v2/$class/*/
for shape in  $(ls /research/d5/gds/rszhu22/SSN_Fitting_current/preprocess/shapenet/$class/);
do
echo $shape
python run_SSN_Fitting_thingi_new_v1_rebuttal_v1_empty_sample_v5_precalc_v2_compare_v1_shapenet.py --dataset shapenet --outdir shapenet  --shape $shape --class_name $class --nepoch 10000 
# exit()
done
done