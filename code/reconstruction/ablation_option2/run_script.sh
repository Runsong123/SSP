################ PE - IS -TS
# shape_list=("120477" "451676" "90276" "thingi_35269" "thingi_36371")
# for shape in  ${shape_list[@]};
# do
# echo $shape
# python run_SSN_Fitting_thingi_new_v1_rebuttal_v1_empty_sample_v5_precalc_v2_compare_ablation_TS_16384.py --dataset thingi --outdir ABLATION_TS_16384  --shape $shape --nepoch 10000 
# done
############## BASE
shape_list=("120477" "451676" "90276" "thingi_35269" "thingi_36371")
for shape in  ${shape_list[@]};
do
echo $shape
python run_SSN_Fitting_thingi_new_v1_rebuttal_v1_empty_sample_v5_precalc_v2_compare_ablation_original.py --dataset thingi --outdir ABLATION_option_2_BASE  --shape $shape --nepoch 10000 
done
################## Add Signed
shape_list=("120477" "451676" "90276" "thingi_35269" "thingi_36371")
for shape in  ${shape_list[@]};
do
echo $shape
python run_SSN_Fitting_thingi_new_v1_rebuttal_v1_empty_sample_v5_precalc_v2_compare_ablation_signed_first.py --dataset thingi --outdir ABLATION_option_2_SIGNED_FIRST  --shape $shape --nepoch 10000 
done
################### Derivative supervision
shape_list=("120477" "451676" "90276" "thingi_35269" "thingi_36371")
for shape in  ${shape_list[@]};
do
echo $shape
python run_SSN_Fitting_thingi_new_v1_rebuttal_v1_empty_sample_v5_precalc_v2_compare_ablation_signed_derivative.py --dataset thingi --outdir ABLATION_option_2_SIGNED_FIRST_Derivative  --shape $shape --nepoch 10000 
r