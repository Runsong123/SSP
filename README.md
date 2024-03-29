<!-- > SSP: Semi-signed prioritized neural fitting for surface reconstruction\\ from unoriented point clouds<br>
> Runsong Zhu¹, Di Kang², Ka-Hei Hui¹, Yue Qian², Shi Qiu¹, Zhen Dong³, Linchao Bao², Pheng-Ann Heng¹, Chi-Wing Fu¹ <br>
> [Project Page](https://runsong123.github.io/SSP/)
¹The Chinese University of Hong Kong + ²Tencent AI Lab + ³Wuhan University.
Under construction ... -->

# SSP: Semi-signed prioritized neural fitting for surface reconstruction from unoriented point clouds (WACV2024)

**[Page](https://runsong123.github.io/SSP/)**, **[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Zhu_SSP_Semi-Signed_Prioritized_Neural_Fitting_for_Surface_Reconstruction_From_Unoriented_WACV_2024_paper.pdf)**, **[Poster](https://runsong123.github.io/SSP/media/wacv24-2527.pdf)**

Runsong Zhu¹*, Di Kang², Ka-Hei Hui¹, Yue Qian², Shi Qiu¹, Zhen Dong³, Linchao Bao², Pheng-Ann Heng¹, Chi-Wing Fu¹.

(*Work partially done during an internship at Tencent AI Lab)

¹The Chinese University of Hong Kong + ²Tencent AI Lab + ³Wuhan University.

<p align="center">
  <a href="">
    <img src="./media/Pipline_order.png" alt="Logo" width="95%">
  </a>
</p>

## Set up
```
pip install -r code/reconstruction/requirements.txt
```


## How to use the code


### Data praparation (calculating the outside regions). 
```
cd code/space_carving
python generate_abc_adapative_add_outside_depth.py
python generate_pcp_adaptive_add_outside_depth.py
python generate_SRB_adaptive_add_outside_depth.py
```
We also provide the processed data ([Here](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155183723_link_cuhk_edu_hk/ErPDv-RZh-lDuCz-BiBN-mwBec97tyjns7wtrfMZKnpckQ?e=vjmv1Z)). Note that, we use the [AdaFit](https://github.com/Runsong123/AdaFit) to calculate the unoriented normals to boost the performance on mentioned datasets in our paper. If you want to test your data, you could generate the normals using [AdaFit](https://github.com/Runsong123/AdaFit) or other methods (e.g., PCA). 

###  Start fitting for given point clouds input
```
cd code/reconstruction
python SSP.py --dataset thingi  --shape 120477 --nepoch 10000 --outdir thingi10K_exps # example
```

## Acknowledgements
Some code snippets are borrowed from [IGR](https://github.com/amosgropp/IGR) and [SAP](https://github.com/autonomousvision/shape_as_points.git) codebases. We thank the authors for releasing their code.

## Citation
If you find our work useful in your research, please cite our paper. 
```
@inproceedings{zhu2024ssp,
  title={SSP: Semi-Signed Prioritized Neural Fitting for Surface Reconstruction From Unoriented Point Clouds},
  author={Zhu, Runsong and Kang, Di and Hui, Ka-Hei and Qian, Yue and Qiu, Shi and Dong, Zhen and Bao, Linchao and Heng, Pheng-Ann and Fu, Chi-Wing},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={3769--3778},
  year={2024}
}
```


