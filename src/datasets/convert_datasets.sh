# Change to your 
OUT_DIR='/Users/kanazawa/projects/tf_datasets/'

# Change to where each dataset directory is:
LSP_DIR='/scratch1/storage/human_datasets/lsp_dataset/'
LSP_EXT_DIR='/scratch1/storage/human_datasets/lsp_extended/'
MPII_DIR='/scratch1/storage/human_datasets/mpii/'

# LSP:
python lsp_to_tfrecords.py --img_directory $LSP_DIR --output_directory $OUT_DIR/lsp
# LSP-extended:
python lsp_to_tfrecords.py --img_directory $LSP_EXT_DIR --output_directory $OUT_DIR/lsp_ext

# MPII:
python mpii_to_tfrecords.py --img_directory $MPII_DIR --output_directory $OUT_DIR/mpii
