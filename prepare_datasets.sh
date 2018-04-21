# ---------------------------
# ----- SET YOUR PATH!! -----
# ---------------------------
# This is where you want all of your tf_records to be saved:
DATA_DIR=/scratch1/storage/hmr_release_files/test_tf_datasets

# This is the directory that contains README.txt
LSP_DIR=/scratch1/storage/human_datasets/lsp_dataset

# This is the directory that contains README.txt
LSP_EXT_DIR=/scratch1/storage/human_datasets/lsp_extended

# This is the directory that contains 'images' and 'annotations'
MPII_DIR=/scratch1/storage/human_datasets/mpii

# This is the directory that contains README.txt
COCO_DIR=/scratch1/storage/coco

# This is the directory that contains README.txt, S1..S8, etc
MPI_INF_3DHP_DIR=/scratch1/storage/mpi_inf_3dhp

## Mosh
# This is the path to the directory that contains neutrSMPL_* directories
MOSH_DIR=/scratch1/storage/human_datasets/neutrMosh
# ---------------------------


# ---------------------------
# Run each command below from this directory. I advice to run each one independently.
# ---------------------------

# ----- LSP -----
python -m src.datasets.lsp_to_tfrecords --img_directory $LSP_DIR --output_directory $DATA_DIR/lsp

# ----- LSP-extended -----
python -m src.datasets.lsp_to_tfrecords --img_directory $LSP_EXT_DIR --output_directory $DATA_DIR/lsp_ext

# ----- MPII -----
python -m src.datasets.mpii_to_tfrecords --img_directory $MPII_DIR --output_directory $DATA_DIR/mpii

# ----- COCO -----
python -m src.datasets.coco_to_tfrecords --data_directory $COCO_DIR --output_directory $DATA_DIR/coco

# ----- MPI-INF-3DHP -----
python -m src.datasets.mpi_inf_3dhp_to_tfrecords --data_directory $MPI_INF_3DHP_DIR --output_directory $DATA_DIR/mpi_inf_3dhp

# ----- Mosh data, for each dataset -----
# CMU:
python -m src.datasets.smpl_to_tfrecords --data_directory $MOSH_DIR --output_directory $DATA_DIR/mocap_neutrMosh --dataset_name 'neutrSMPL_CMU'

# H3.6M:
python -m src.datasets.smpl_to_tfrecords --data_directory $MOSH_DIR --output_directory $DATA_DIR/mocap_neutrMosh --dataset_name 'neutrSMPL_H3.6'

# jointLim:
python -m src.datasets.smpl_to_tfrecords --data_directory $MOSH_DIR --output_directory $DATA_DIR/mocap_neutrMosh --dataset_name 'neutrSMPL_jointLim'
