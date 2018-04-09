""" 
Sets default args

Note all data format is NHWC because slim resnet wants NHWC.
"""

from absl import flags
import os.path as osp
curr_path = osp.dirname(osp.abspath(__file__))
model_dir = osp.join(curr_path, '..', 'models')
if not osp.exists(model_dir):
    print('Fix path to models/')
    import ipdb
    ipdb.set_trace()
SMPL_MODEL_PATH = osp.join(model_dir, 'neutral_smpl_with_cocoplus_reg.pkl')
SMPL_FACE_PATH = osp.join(curr_path, '../src/tf_smpl', 'smpl_faces.npy')

# Default model path
PRETRAINED_MODEL = osp.join(model_dir, 'model.ckpt-667589')

flags.DEFINE_string('smpl_model_path', SMPL_MODEL_PATH,
                    'path to the neurtral smpl model')
flags.DEFINE_string('smpl_face_path', SMPL_FACE_PATH,
                    'path to  smpl mesh faces (for easy rendering)')
flags.DEFINE_string('load_path', PRETRAINED_MODEL, 'path to trained model')
flags.DEFINE_integer('batch_size', 1,
                     'Input image size to the network after preprocessing')

# Don't change if testing:
flags.DEFINE_integer('img_size', 224,
                     'Input image size to the network after preprocessing')
flags.DEFINE_string('data_format', 'NHWC', 'Data format')
flags.DEFINE_integer('num_stage', 3, '# of times to iterate regressor')
flags.DEFINE_string('model_type', 'resnet_fc3_dropout',
                    'What kind of networks to use')
flags.DEFINE_string(
    'joint_type', 'cocoplus',
    'cocoplus (19 keypoints) or lsp 14 keypoints, returned by SMPL')
