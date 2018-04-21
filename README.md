# End-to-end Recovery of Human Shape and Pose

Angjoo Kanazawa, Michael J. Black, David W. Jacobs, Jitendra Malik
CVPR 2018

[Project Page](https://akanazawa.github.io/hmr/)
![Teaser Image](https://akanazawa.github.io/hmr/resources/images/teaser.png)

### Requirements
- Python 2.7
- [TensorFlow](https://www.tensorflow.org/) tested on version 1.3

### Installation

#### Setup virtualenv
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

On images that are not tightly cropped, you can run
[openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and supply
its output json (run it with `--write_json` option).
When json_path is specified, the demo will compute the right scale and bbox center to run HMR:
```
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
```
(The demo only runs on the most confident bounding box, see `src/util/openpose.py:get_bbox`)

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
  booktitle={Computer Vision and Pattern Regognition (CVPR)},
  year={2018}
}
```
