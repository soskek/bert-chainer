# Chainer implementation of Google AI's BERT model with a script to load Google's pre-trained models

This repository contains a Chainer reimplementation of [Google's TensorFlow repository for the BERT model](https://github.com/google-research/bert) for the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.

This implementation can load any pre-trained TensorFlow checkpoint for BERT (in particular [Google's pre-trained models](https://github.com/google-research/bert)) and a conversion script is provided (see [below](#loading-a-tensorflow-checkpoint-eg-googles-pre-trained-models)).

In the current implementation, we can

- build BertModel and load pre-trained checkpoints from TensorFlow
- apply it to typical classification tasks, i.e., finetuning and evaluation (see [below](#fine-tuning-with-bert-running-the-examples))

WIP and TODO:

- implement minor issues (e.g. warmup Adam for finetuning)
- implement the SQuAD QA example
- implement `extract_features.py` example
- pretrain BertModel in a new corpus, with multiGPU
- test multilingual models (https://github.com/google-research/bert/blob/master/multilingual.md)

This README follows the great README of [PyTorch's BERT repository](https://github.com/huggingface/pytorch-pretrained-BERT) by [the huggingface team](https://github.com/huggingface).

## Loading a TensorFlow checkpoint (e.g. [Google's pre-trained models](https://github.com/google-research/bert#pre-trained-models))

You can convert any TensorFlow checkpoint for BERT (in particular [the pre-trained models released by Google](https://github.com/google-research/bert#pre-trained-models)) in a Chainer save file by using the [`convert_tf_checkpoint_to_chainer.py`](convert_tf_checkpoint_to_chainer.py) script.

This script takes as input a TensorFlow checkpoint (three files starting with `bert_model.ckpt`) and creates a Chainer model (npz file) for this configuration, so that we can load the models using `chainer.serializers.load_npz()` by Chainer. (see examples in `run_classifier.py`)

You only need to run this conversion script **once** to get a Chainer model. You can then disregard the TensorFlow checkpoint (the three files starting with `bert_model.ckpt`) but be sure to keep the configuration file (`bert_config.json`) and the vocabulary file (`vocab.txt`) as these are needed for the Chainer model too.

To run this specific conversion script you will need to have TensorFlow and Chainer installed (`pip install tensorflow`). The rest of the repository only requires Chainer.

Here is an example of the conversion process for a pre-trained `BERT-Base Uncased` model:

```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12

python convert_tf_checkpoint_to_chainer.py \
  --tf_checkpoint_path $BERT_BASE_DIR/bert_model.ckpt \
  --npz_dump_path $BERT_BASE_DIR/arrays_bert_model.ckpt.npz
```

You can download Google's pre-trained models for the conversion [here](https://github.com/google-research/bert#pre-trained-models).

## Chainer models for BERT

We included two Chainer models in this repository that you will find in [`modeling.py`](modeling.py):

- `BertModel` - the basic BERT Transformer model
- `BertClassifier` - the BERT model with a sequence classification head on top

Here are some details on each class.

### 1. `BertModel`

`BertModel` is the basic BERT Transformer model with a layer of summed token, position and sequence embeddings followed by a series of identical self-attention blocks (12 for BERT-base, 24 for BERT-large).

The inputs and output are **identical to the TensorFlow model inputs and outputs**.

We detail them here. This model takes as inputs:

- `input_ids`: an int array of shape [batch_size, sequence_length] with the word token indices in the vocabulary (see the tokens preprocessing logic in the scripts `run_classifier.py`), and
- `token_type_ids`: an optional int array of shape [batch_size, sequence_length] with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
- `attention_mask`: an optional array of shape [batch_size, sequence_length] with indices selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max input sequence length in the current batch. It's the mask that we typically use for attention when a batch has varying length sentences.

This model outputs a tuple composed of:

- `all_encoder_layers`: a list of torch.FloatTensor of size [batch_size, sequence_length, hidden_size] which is a list of the full sequences of hidden-states at the end of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), and
- `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a classifier pretrained on top of the hidden state associated to the first character of the input (`CLF`) to train on the Next-Sentence task (see BERT's paper).

An example on how to use this class is given in the `extract_features.py` script which can be used to extract the hidden states of the model for a given input.

### 2. `BertClassifier`

`BertClassifier` is a fine-tuning model that includes `BertModel` and a sequence-level (sequence or pair of sequences) classifier on top of the `BertModel`.

The sequence-level classifier is a linear layer that takes as input the last hidden state of the first character in the input sequence (see Figures 3a and 3b in the BERT paper).

An example on how to use this class is given in the `run_classifier.py` script which can be used to fine-tune a single sequence (or pair of sequence) classifier using BERT, for example for the MRPC task.

## Installation, requirements, test

This code was tested on Python 3.5+. The requirements are:

- Chainer
- progressbar2

To install the dependencies:

```bash
pip install -r ./requirements.txt
```

## Fine-tuning with BERT: running the examples

We showcase the same examples as [the original implementation](https://github.com/google-research/bert/): fine-tuning a sequence-level classifier on the MRPC classification corpus.

### Prepare the dataset of GLUE

Before running theses examples you should download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`.

```bash
wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/becd574dd938f045ea5bd3cb77d1d506541b5345/download_glue_data.py
python download_glue_data.py
export GLUE_DIR=./glue_data
```

### Prepare the pretrained BERT model

Please also download the `BERT-Base`
checkpoint, unzip it to some directory `$BERT_BASE_DIR`, and convert it to its Chainer version as explained in the previous section.

```bash
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
export BERT_BASE_DIR=./uncased_L-12_H-768_A-12
python convert_tf_checkpoint_to_chainer.py \
  --tf_checkpoint_path $BERT_BASE_DIR/bert_model.ckpt \
  --npz_dump_path $BERT_BASE_DIR/arrays_bert_model.ckpt.npz
```

### Train and evaluate

This example code fine-tunes `BERT-Base` on the Microsoft Research Paraphrase
Corpus (MRPC) corpus and runs in less than several minutes on a single Tesla P100.

```shell
python run_classifier.py \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/MRPC/ \
  --vocab_file $BERT_BASE_DIR/vocab.txt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --init_checkpoint $BERT_BASE_DIR/arrays_bert_model.ckpt.npz \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/mrpc_output/
```

Our test ran on a few seeds with [the original implementation hyper-parameters](https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks) gave evaluation results between 82 and 87.
