# shape_list=("120477" "451676" "90276" "thingi_35269" "thingi_36371")
shape_list=("daratech" "anchor" "dc" "gargoyle" "lord_quas")
for shape in  ${shape_list[@]};
do
echo $shape
python run_SSN_Fitting_thingi_new_v1_rebuttal_v1_empty_sample_v5_precalc_v2_compare_ablation_TS_16384.py --dataset deep_geometric_prior_data --outdir ABLATION_TS_16384  --shape $shape --nepoch 10000 
done
# shape_list=("anchor" "daratech" "dc" "gargoyle" "lord_quas")
# for shape in  ${shape_list[@]};
# do
# echo $shape
# python run_SSN_Fitting_thingi_new_v1_rebuttal_v1_empty_sample_v5_precalc_v2_compare_famous_evaluate.py --dataset deep_geometric_prior_data --outdir empty_uniform_v5_precalc_v2  --shape $shape --nepoch 5000 --saveobj /research/d5/gds/rszhu22/SSN_Fitting_current/code_v8/reconstruction/challenging 
# done