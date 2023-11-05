# ############## TS
# shape_list=("120477" "451676" "90276" "thingi_35269" "thingi_36371")
# for shape in  ${shape_list[@]};
# do
# echo $shape
# python run_SSN_Fitting_thingi_new_v1_rebuttal_v1_empty_sample_v5_precalc_v2_compare_ablation_TS.py --dataset thingi --outdir ABLATION_TS  --shape $shape --nepoch 10000 
# done
# ################## signed
# shape_list=("120477" "451676" "90276" "thingi_35269" "thingi_36371")
# for shape in  ${shape_list[@]};
# do
# echo $shape
# python run_SSN_Fitting_thingi_new_v1_rebuttal_v1_empty_sample_v5_precalc_v2_compare_ablation_signed.py --dataset thingi --outdir ABLATION_SIGNED  --shape $shape --nepoch 10000 
# done
# ################### Derivative supervision
# shape_list=("120477" "451676" "90276" "thingi_35269" "thingi_36371")
# for shape in  ${shape_list[@]};
# do
# echo $shape
# python run_SSN_Fitting_thingi_new_v1_rebuttal_v1_empty_sample_v5_precalc_v2_compare_ablation_derivative_supervision.py --dataset thingi --outdir ABLATION_DS  --shape $shape --nepoch 10000 
# done
################### PE
shape_list=("120477" "451676" "90276" "thingi_35269" "thingi_36371")
for shape in  ${shape_list[@]};
do
echo $shape
python run_SSN_Fitting_thingi_new_v1_rebuttal_v1_empty_sample_v5_precalc_v2_compare_ablation_PE.py --dataset thingi --outdir ABLATION_PE_GPU_check  --shape $shape --nepoch 10000 
done