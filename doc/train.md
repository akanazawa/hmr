## Pre-reqs

### Download required models

1. Download the mean SMPL parameters (initialization)
```
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/neutral_smpl_mean_params.h5
```

Store this inside `hmr/models/`, along with the neutral SMPL model
(`neutral_smpl_with_cocoplus_reg.pkl`).


2. Download the pre-trained resnet-50 from
[Tensorflow](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained)
```
wget http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz && tar -xf resnet_v2_50_2017_04_14.tar.gz
```

3. In `src/do_train.sh`, replace the path of `PRETRAINED` to the path of this model (`resnet_v2_50.ckpt`).

### Download datasets.
Download these datasets somewhere.

- [LSP](http://sam.johnson.io/research/lsp_dataset.zip) and [LSP extended](http://sam.johnson.io/research/lspet_dataset.zip)
- [COCO](http://cocodataset.org/#download) we used 2014 Train. You also need to
  install the [COCO API](https://github.com/cocodataset/cocoapi) for python.
- [MPII](http://human-pose.mpi-inf.mpg.de/#download)
- [MPI-INF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/)

For Human3.6M, download the pre-computed tfrecords [here](https://drive.google.com/open?id=1tquavoVWSdGeOn9P6zwoffIMoCRElzEO).
Note that this is 11GB! I advice you do this in a directly outside of the HMR code base.


If you use the datasets above, please consider citing their original papers.

## Mosh Data. 
We provide the MoShed data using the neutral SMPL model.
Please note that usage of this data is for [**non-comercial scientific research only**](http://mosh.is.tue.mpg.de/data_license).

If you use any of the MoSh data, please cite: 
```
article{Loper:SIGASIA:2014,
  title = {{MoSh}: Motion and Shape Capture from Sparse Markers},
  author = {Loper, Matthew M. and Mahmood, Naureen and Black, Michael J.},
  journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
  volume = {33},
  number = {6},
  pages = {220:1--220:13},
  publisher = {ACM},
  address = {New York, NY, USA},
  month = nov,
  year = {2014},
  url = {http://doi.acm.org/10.1145/2661229.2661273},
  month_numeric = {11}
}
```

[Download link to MoSh](https://drive.google.com/file/d/1b51RMzi_5DIHeYh2KNpgEs8LVaplZSRP/view?usp=sharing)

## TFRecord Generation

All the data has to be converted into TFRecords and saved to a `DATA_DIR` of
your choice.

1. Make `DATA_DIR` where you will save the tf_records. For ex:
```
mkdir ~/hmr/tf_datasets/
```

2. Edit `prepare_datasets.sh`, with paths to where you downloaded the datasets,
and set `DATA_DIR` to the path to the directory you just made.

3. From the root HMR directly (where README is), run `prepare_datasets.sh`, which calls the tfrecord conversion scripts:
```
sh prepare_datasets.sh
```

This takes a while! If there is an issue consider running line by line.

4. Move the downloaded human36m tf_records `tf_records_human36m.tar.gz` into the
`data_dir`:
```
tar -xf tf_records_human36m.tar.gz
```

5. In `do_train.sh` and/or `src/config.py`, set `DATA_DIR` to the path where you saved the
tf_records.


## Training
Finally we can start training!
A sample training script (with parameters used in the paper) is in
`do_train.sh`.

Update the path to  in the beginning of this script and run:
```
sh do_train.sh
```

The training write to a log directory that you can specify.
Setup tensorboard to this directory to monitor the training progress like so:
![Teaser Image](https://akanazawa.github.io/hmr/resources/images/tboard_ex.png)

It's important to visually monitor the training! Make sure that the images
loaded look right.


## Evaluation
Provided is an evaluation code for Human3.6M. It uses the test tf_records,
provided with the training tf_records available above and [here (11GB)](https://drive.google.com/open?id=1tquavoVWSdGeOn9P6zwoffIMoCRElzEO).

To evaluate a model, run
```
python -m src.benchmark.evaluate_h36m --batch_size=500
--load_path=<model_to_eval.ckpt> --tfh36m_dir <path to tf_records_human36m_wjoints/>
```
for example for the provided model, use:
```
python -m src.benchmark.evaluate_h36m --batch_size=500
--load_path=models/model.ckpt-667589 --tfh36m_dir <path to tf_records_human36m_wjoints/>
```

This writes intermediate output to a temp directory, which you can specify by pred_dir
With the provided model, this outputs errors per action and overall MPE for P1
(corresponding to Table 2 in paper -- this retrained model gets slightly lower
MPJPE and a comparable PA-MPJPE):
```
MPJPE: 86.20, PA-MPJPE: 58.47, Median: 79.47, PA-Median: 52.95
```

Run it with `--vis` option, it visualizes the top/worst 30 results. 
