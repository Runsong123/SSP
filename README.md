<!-- > SSP: Semi-signed prioritized neural fitting for surface reconstruction\\ from unoriented point clouds<br>
> Runsong Zhu¹, Di Kang², Ka-Hei Hui¹, Yue Qian², Shi Qiu¹, Zhen Dong³, Linchao Bao², Pheng-Ann Heng¹, Chi-Wing Fu¹ <br>
> [Project Page](https://runsong123.github.io/SSP/)
¹The Chinese University of Hong Kong + ²Tencent AI Lab + ³Wuhan University.
Under construction ... -->


# SSP: Semi-signed prioritized neural fitting for surface reconstruction from unoriented point clouds (WACV2024)

**[Project Page](https://runsong123.github.io/SSP/) | [Arxiv](https://arxiv.org/abs/2206.06715) |  [Poster](https://runsong123.github.io/SSP/) |**

Runsong Zhu¹, Di Kang², Ka-Hei Hui¹, Yue Qian², Shi Qiu¹, Zhen Dong³, Linchao Bao², Pheng-Ann Heng¹, Chi-Wing Fu¹.

¹The Chinese University of Hong Kong + ²Tencent AI Lab + ³Wuhan University.


## Set up
```
pip install -r code/reconstruction/requirements.txt
```


## How to use the code


### Data praparation (calculating the outside region). 
```
cd code/space_carving
python generate_abc_adapative_add_outside_depth.py
python generate_pcp_adaptive_add_outside_depth.py
python generate_SRB_adaptive_add_outside_depth.py
```
We also provide the processed data ([Here](https://runsong123.github.io/SSP/)). You can download it and then place it in ```./data/```. Note that, we use the [AdaFit](https://github.com/Runsong123/AdaFit) to calculate the unoriented normals to boost the performance on mentioned datasets in our paper. If you want to test your data, you could generate the normals using [AdaFit](https://github.com/Runsong123/AdaFit) or other methods (e.g., PCA). 

###  Start fitting for given point cloud input
```
cd code/reconstruction
python SSP.py --dataset thingi  --shape 120477 --nepoch 10000 --outdir thingi10K_exps # example
```

## Acknowledgements
This project is built upon [IGR](https://github.com/amosgropp/IGR) and [SAP](https://github.com/autonomousvision/shape_as_points.git) codebases. We thank the authors for releasing their code.

## Citation
If you find our work useful in your research, please cite our paper. 
```
@article{zhu2022semi,
  title={Semi-signed neural fitting for surface reconstruction from unoriented point clouds},
  author={Zhu, Runsong and Kang, Di and Hui, Ka-Hei and Qian, Yue and Zhe, Xuefei and Dong, Zhen and Bao, Linchao and Fu, Chi-Wing},
  journal={arXiv preprint arXiv:2206.06715},
  year={2022}
}
```


