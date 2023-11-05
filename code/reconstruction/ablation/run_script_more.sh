############## signed distance
shape_list=("120477" "451676" "90276" "thingi_35269" "thingi_36371")
for shape in  ${shape_list[@]};
do
echo $shape
python run_SSN_Fitting_thingi_new_v1_rebuttal_v1_empty_sample_v5_precalc_v2_compare_v1_direclty_signed_distance.py --dataset thingi --outdir ABLATION_signed_distance_supervision  --shape $shape --nepoch 10000 
done
################## signed
# shape_list=("120477" "451676" "90276" "thingi_35269" "thingi_36371")
# for shape in  ${shape_list[@]};
# do
# echo $shape
# python run_SSN_Fitting_thingi_new_v1_rebuttal_v1_empty_sample_v5_precalc_v2_compare_ablation_signed.py --dataset thingi --outdir ABLATION_SIGNED  --shape $shape --nepoch 10000 
# done
