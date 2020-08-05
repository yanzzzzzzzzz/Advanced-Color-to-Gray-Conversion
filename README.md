# Advanced-Color-to-Gray-Conversion
An implementation of "Song, Yibing & Bao, Linchao & Xu, Xiaobin & Yang, Qingxiong. (2013). Decolorization: Is rgb2gray() out?. SIGGRAPH Asia 2013 Technical Briefs, SA "

## Requirement
* python3 (or higher)
* opencv 3.0

install package 

* numpy
* matplotlib

## Usage
```bash
$ python main.py --mode <c: conventional; a: advanced> -i <input image folder> -o <output directory>

# for example to process advanced grayscale 
$ python main.py
```

## Output
The program will output:
* Conventional mode
  * RGB image convert to grayscale image using``` Y = 0.299 * R + 0.587 * G + 0.114 * B``` formula
* Advanced mode
  * RGB image convert to grayscale image using [Decolorization: Is rgb2gray() Out?](https://ybsong00.github.io/siga13tb/siga13tb_final.pdf) method to select weight
  * Record weight and vote in save file name
* Processing all image in input folder
* Save process result in output directory

see more detail in [Report](https://github.com/yanzzzzzzzzz/Advanced-Color-to-Gray-Conversion/blob/master/Report.pdf)
