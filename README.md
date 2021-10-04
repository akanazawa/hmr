# End-to-end Recovery of Human Shape and Pose

Angjoo Kanazawa, Michael J. Black, David W. Jacobs, Jitendra Malik
CVPR 2018

[Project Page](https://akanazawa.github.io/hmr/)
![Teaser Image](https://akanazawa.github.io/hmr/resources/images/teaser.png)

### Requirements
- Python 2.7
- [TensorFlow](https://www.tensorflow.org/) tested on version 1.3, demo alone runs with TF 1.12

### Installation

#### Linux Setup with virtualenv
```
virtualenv venv_hmr
source venv_hmr/bin/activate
pip install -U pip
deactivate
source venv_hmr/bin/activate
pip install -r requirements.txt
```
#### Install TensorFlow
With GPU:
```
pip install tensorflow-gpu==1.3.0
```
Without GPU:
```
pip install tensorflow==1.3.0
```

### Windows Setup with python 3 and Anaconda
This is only partialy tested.
```
conda env create -f hmr.yml
```
#### if you need to get chumpy 
https://github.com/mattloper/chumpy/tree/db6eaf8c93eb5ae571eb054575fb6ecec62fd86d


### Demo

1. Download the pre-trained models
```
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz && tar -xf models.tar.gz
```

2. Run the demo
```
python -m demo --img_path data/coco1.png
python -m demo --img_path data/im1954.jpg
```

Images should be tightly cropped, where the height of the person is roughly 150px.
On images that are not tightly cropped, you can run
[openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and supply
its output json (run it with `--write_json` option).
When json_path is specified, the demo will compute the right scale and bbox center to run HMR:
```
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
```
(The demo only runs on the most confident bounding box, see `src/util/openpose.py:get_bbox`)

### Batch Demo
Author [Lotayou](github.com/Lotayou)

To run the demo on batch images, do 
```
python -m demo_batch --img_path data/small --json_path data/small_keypoints_18 --result_path results
```

For more details, please check `demo_batch.py`. 

__Note__: Make sure image filename and json filename come in pairs, like `img.png` and `img_keypoints.json`.


### Webcam Demo (thanks @JulesDoe!)
1. Download pre-trained models like above.
2. Run webcam Demo
2. Run the demo
```
python -m demo --img_path data/coco1.png
python -m demo --img_path data/im1954.jpg
```

### Training code/data
Please see the [doc/train.md](https://github.com/akanazawa/hmr/blob/master/doc/train.md)!

### Citation
If you use this code for your research, please consider citing:
```
@inProceedings{kanazawaHMR18,
  title={End-to-end Recovery of Human Shape and Pose},
  author = {Angjoo Kanazawa
  and Michael J. Black
  and David W. Jacobs
  and Jitendra Malik},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
```

### Opensource contributions
[russoale](https://github.com/russoale/) has created a Python 3 version with TF 2.0: https://github.com/russoale/hmr2.0

[Dawars](https://github.com/Dawars) has created a docker image for this project: https://hub.docker.com/r/dawars/hmr/

[MandyMo](https://github.com/MandyMo) has implemented a pytorch version of the repo: https://github.com/MandyMo/pytorch_HMR.git

[Dene33](https://github.com/Dene33) has made a .ipynb for Google Colab that takes video as input and returns .bvh animation!
https://github.com/Dene33/video_to_bvh 

<img alt="bvh" src="https://i.imgur.com/QxML83b.gif" /><img alt="" src="https://i.imgur.com/vfge7DS.gif" />
<img alt="bvh2" src=https://i.imgur.com/UvBM1gv.gif />

[layumi](https://github.com/layumi) has added a 2D-to-3D color mapping function to the final obj: https://github.com/layumi/hmr
<img width=200px src=https://github.com/layumi/hmr/blob/master/demo.png />

I have not tested them, but the contributions are super cool! Thank you!!
Let me know if you have any mods that you would like to be added here!


