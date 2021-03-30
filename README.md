---
title: README
date: 2021-3-29 13:16:23
categories: README副本
tags: 
- cuda inference
---
----------
&ensp;&ensp;This repo is use cpu or gpu to inference yolo-v3-tiny model,and the main test environment is:
***

    Ubuntu 18.04 LTS
    GPU 2080Ti
    CUDA 10.2
*** 
&ensp;&ensp;This case provide yolo-v3-tiny model inference option  with cpu and gpu,also can import others extension.And use this case to inference :
***
    gcc main.c -o inference -lm
    sudo ./inference -gpu ./test.jpg 
***
&ensp;&ensp;Or use "sudo ./inference -cpu ./test.jpg" to use cpu inference.


