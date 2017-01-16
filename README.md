# imageAlign

![image](./data/photo.jpg)

## 1. Introduction
The goal of image alignment is to align a template image ***T*** to an input image ***I***. 
This demo contains four classical image alignment algorithms. 
We implemented these algorithms in C++ language using [OpenCV](https://github.com/opencv/opencv/tree/3.1.0) library in version 3.1.0. 
With the aim of algorithm verification, we did not take the efficiency seriously, so the code can be achieve in some more efficient ways.

## 2. Implementation
### 2.1 Environment
If you have not installed the [OpenCV 3.1.0 ](https://github.com/opencv/opencv/tree/3.1.0), please set you OpenCV build path to the `OpenCV_DIR` in CMakeLists.txt.
Such as:

    %your directory%/opencv/build

### 2.2 Build & Run
In the root directory of the image alignment demo, run the following comments:

	$ mkdir build && cd build
	$ cmake ..
	$ make

After building, there will be 4 executable files in `/bin`, that we can just run.
Such as:

	../bin/test_additive

## Reference

- [Lucas-Kanade 20 Years On: A Unifying Framework](http://www.ncorr.com/download/publications/bakerunify.pdf)
- [Equivalence and Efficiency of Image Alignment Algorithms](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=990652)
- [Image Alignment Algorithms(Code Project)](https://www.codeproject.com/Articles/24809/Image-Alignment-Algorithms)