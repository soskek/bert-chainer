import argparse
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--tf_checkpoint_path",
                    required=True,
                    help="Path the TensorFlow checkpoint path."
                    " e.g. ./uncased_L-12_H-768_A-12/bert_model.ckpt")
parser.add_argument("--npz_dump_path",
                    required=True,
                    help="Path to the output Chainer model (arrays in npz)."
                    "e.g. ./uncased_L-12_H-768_A-12/arrays_bert_model.ckpt.npz")
args = parser.parse_args()

# LOAD_CKPT_PATH = './bert_model.ckpt'
# SAVE_NPZ_PATH = './arrays_bert_model.ckpt.npz'
LOAD_CKPT_PATH = args.tf_checkpoint_path
SAVE_NPZ_PATH = args.npz_dump_path

# load
sess = tf.Session()
new_saver = tf.train.import_meta_graph(
    LOAD_CKPT_PATH + '.meta')
what = new_saver.restore(
    sess, LOAD_CKPT_PATH)
all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

arrays = dict()
for v in all_vars:
    v_ = sess.run(v)  # np.ndarray
    arrays[v.name] = v_

# rename arrays for alignment between this and original
new_arrays = dict()
for name, v in arrays.items():
    new_name = name
    new_name = new_name.rstrip(':0')

    # embeddings
    new_name = new_name.replace(
        '/embeddings/LayerNorm/', '/post_embedding_ln/')
    new_name = new_name.replace('/embeddings/', '/')
    new_name = new_name.replace('_embeddings', '_embeddings/W')

    # attention
    new_name = new_name.replace('/attention/output/', '/attention_output/')
    new_name = new_name.replace(
        'attention_output/dense', 'attention_output_dense')
    new_name = new_name.replace('/attention/self/', '/attention/')
    new_name = new_name.replace('/LayerNorm/', '_ln/')

    # attention
    new_name = new_name.replace(
        '/intermediate/dense/', '/intermediate_dense1/')
    new_name = new_name.replace('/output/dense/', '/intermediate_dense2/')
    new_name = new_name.replace('/attention/self/', '/attention/')
    new_name = new_name.replace('/output_ln/', '/intermediate_ln/')

    # pooler
    new_name = new_name.replace('/pooler/dense/', '/pooler/')

    # all
    new_name = new_name.replace('bias', 'b')
    new_name = new_name.replace('kernel', 'W')

    # TensorFlow's dense matrix is transposed version of that in Chainer
    if 'kernel' in name:
        v = v.T

    new_arrays[new_name] = v

    """ ignore
    cls/predictions/output_bias:0    (30522,)
    cls/predictions/transform/LayerNorm/beta:0       (768,)
    cls/predictions/transform/LayerNorm/gamma:0      (768,)
    cls/predictions/transform/dense/bias:0   (768,)
    cls/predictions/transform/dense/kernel:0         (768, 768)
    cls/seq_relationship/output_bias:0       (2,)
    cls/seq_relationship/output_weights:0    (2, 768)
    """


# save
# np.savez(SAVE_NPZ_PATH, **arrays)
np.savez(SAVE_NPZ_PATH, **new_arrays)

loaded_arrays = np.load(SAVE_NPZ_PATH)

# print
for n, a in sorted(loaded_arrays.items()):
    print(n, '\t', a.shape)
print(SAVE_NPZ_PATH)

# print("====================================")
# for n, a in sorted(arrays.items()):
#     print(n, '\t', a.shape)
