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



---


# Trying to Run it on Colab page (As A Forker!):

I have made [this colab page](https://colab.research.google.com/github/soheilpaper/-tft-2.4-ili9341-STM32/blob/master/3D_pose_Estimation/Demo_MeshRCNN.ipynb) and get these bugs:
```
for some common reasons and solutions.  Include the entire stack trace
above this error message when asking for help.
Traceback (most recent call last):
  File "/usr/lib/python2.7/runpy.py", line 174, in _run_module_as_main
    "__main__", fname, loader, pkg_name)
  File "/usr/lib/python2.7/runpy.py", line 72, in _run_code
    exec code in run_globals
  File "/content/hmr/demo.py", line 27, in <module>
    import tensorflow as tf
  File "/tensorflow-1.15.2/python3.7/tensorflow/__init__.py", line 99, in <module>
    from tensorflow_core import *
  File "/tensorflow-1.15.2/python3.7/tensorflow_core/__init__.py", line 28, in <module>
    from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
  File "/tensorflow-1.15.2/python3.7/tensorflow/__init__.py", line 50, in __getattr__
    module = self._load()
  File "/tensorflow-1.15.2/python3.7/tensorflow/__init__.py", line 44, in _load
    module = _importlib.import_module(self.__name__)
  File "/usr/lib/python2.7/importlib/__init__.py", line 37, in import_module
    __import__(name)
  File "/tensorflow-1.15.2/python3.7/tensorflow_core/python/__init__.py", line 49, in <module>
    from tensorflow.python import pywrap_tensorflow
  File "/tensorflow-1.15.2/python3.7/tensorflow_core/python/pywrap_tensorflow.py", line 74, in <module>
    raise ImportError(msg)
ImportError: Traceback (most recent call last):
  File "/tensorflow-1.15.2/python3.7/tensorflow_core/python/pywrap_tensorflow.py", line 58, in <module>
    from tensorflow.python.pywrap_tensorflow_internal import *
  File "/tensorflow-1.15.2/python3.7/tensorflow_core/python/pywrap_tensorflow_internal.py", line 28, in <module>
    _pywrap_tensorflow_internal = swig_import_helper()
  File "/tensorflow-1.15.2/python3.7/tensorflow_core/python/pywrap_tensorflow_internal.py", line 24, in swig_import_helper
    _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
ImportError: dynamic module does not define init function (init_pywrap_tensorflow_internal)


Failed to load the native TensorFlow runtime.

See https://www.tensorflow.org/install/errors

for some common reasons and solutions.  Include the entire stack trace
above this error message when asking for help.
```



and asked [here](https://stackoverflow.com/questions/67342936/failed-to-load-the-native-tensorflow-runtime-colab-error) and on [the Issue part of the main project on GitHub site.](https://github.com/akanazawa/hmr/issues/155) :


![image](https://user-images.githubusercontent.com/6679151/116770975-fa6b9600-aa5c-11eb-9f15-67e51e634114.png)






