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
- [MPI-INF-3DHP](http://human-pose.mpi-inf.mpg.de/#download)

For Human3.6M, download the pre-computed tfrecords [here](https://drive.google.com/file/d/14RlfDlREouBCNsR1QGDP0qpOUIu5LlV5/view?usp=sharing).
Note that this is 9.1GB! I advice you do this in a directly outside of the HMR code base.
```
wget https://angjookanazawa.com/cachedir/hmr/tf_records_human36m.tar.gz
```

If you use these datasets, please consider citing them.

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

