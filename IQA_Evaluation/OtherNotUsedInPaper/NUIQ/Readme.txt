# 修改demo.m中的路径（图片）与test_model.m中的路径（结果）后直接运行即可
# demo.m中需修改groups值为图片数量，并更改folder路径与files的格式

# NUIQ Metric released.
This is the code of the implementation of the  No-reference (NR) Underwater Image Quality metric (NUIQ). If you use our code or dataset for academic purposes, please consider citing our paper. Thanks.

# Requirment
Matlab.

# Noting
Our NUIQ Metric was trained to predict the rank of different enhanced results associated with the same raw underwater image. Therefore, the NUIQ scores cannot be used to compare the quality of enhanced images with different contents.
Do not test on single image!

## Usage
First of all, you need to modify the true path. Then, Training and testing.

## Training
1. Run the train_model.m

## Testing
1. Run the demo.m

If you find this work useful for you. Please cite:
@ARTICLE{9749233,
author={Jiang, Qiuping and Gu, Yuese and Li, Chongyi and Cong, Runmin and Shao, Feng},
journal={IEEE Transactions on Circuits and Systems for Video Technology},
title={Underwater Image Enhancement Quality Evaluation: Benchmark Dataset and Objective Metric},
year={2022},
volume={},
number={},
pages={1-1},
doi={10.1109/TCSVT.2022.3164918}
}

If you have any problem of our program, please feel free to contact with the authors:
jiangqiuping@nbu.edu.cn, 805682724@qq.com

========================================================================
 
-----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------
 
Copyright (c) 2022 Ningbo University (NBU)
All rights reserved.
 
Permission is hereby granted, without written agreement and without license or royalty fees, to use, copy, 
modify, and distribute this dataset and code for any purpose, provided that the copyright notice in its 
entirety appear in all copies of this database and code, the research is to be cited in the bibliography as:
 
1)
 
IN NO EVENT SHALL NBU BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR 
CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS DATASET AND CODE, EVEN IF NBU 
HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
NBU SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE DATASET AND 
CODE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, NBU HAS NO OBLIGATION TO PROVIDE 
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 
-----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------%
 
========================================================================
