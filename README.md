# MetaHTR

Implementation of the paper titled "MetaHTR: Towards Writer-Adaptive Handwritten Text
Recognition" by Bhunia et al. ([Arxiv link](https://arxiv.org/abs/2104.01876)).

![](img/metahtr.png)

The central idea is to use model-agnostic meta-learning (MAML) to make HTR models more
adaptive to unseen writers. The figure above shows the optimization process, involving
an inner loop (left) for writer-specific adaptation and an outer loop (right) for
learning underlying "meta-knowledge" that optimizes for weights that can be rapidly
adapted to novel writers.

MetaHTR [[1]](#References) can be applied to most HTR models, due to its model-agnostic
nature. We include the following state-of-the-art models here:

- SAR [[2]](#References): LSTM-based architecture making use of a 2D attention module.
  Consists of a ResNet backbone, LSTM encoder, LSTM encoder, 2D attention module.
- FPHTR [[3]](#References): Transformer-based architecture. Consists of a ResNet
  backbone combined with a Transformer decoder ([original
  repo](https://github.com/tobiasvanderwerff/full-page-handwriting-recognition))

The idea is to apply MetaHTR to a trained HTR model. It may also be possible to
train a model using MetaHTR from randomly initialized weights, but in my experience
this does not work very well (and takes quite some time to run). More realistically,
MetaHTR is applied for 10-40 epochs on a trained model.

## Dataset
IAM (TODO)


## How to install
Tested using Python 3.8.

```shell
# Clone this repository.
git clone https://github.com/tobiasvanderwerff/MetaHTR.git
cd MetaHTR

git submodule update --init       # initialize submodules
python3 -m venv env               # create a new virtual environment
source env/bin/activate           # activate the environment
pip install -r requirements.txt   # install requirements
pip install -e .                  # install this repo as a package
pip install -e htr                # install submodule containing base models
```

## Example
Training MetaHTR using FPHTR as base model:
```shell
python main.py ...
```
TODO

## Using other base models
Currently, only the two base models mentioned above are supported. However, MetaHTR
and MAML in particular can be applied to most architectures without any real
modifications. In other words, it should be possible to use MetaHTR on your HTR
model of choice, but currently this is not supported yet in the code.

TODO

## Results
TODO

## TODO
- Move finetuning model to separate branch and then decide what to do with it
- Maybe rename main to train.py
- remove (redundant) commented code

## References
[1] Bhunia, Ayan Kumar, et al. "Metahtr: Towards writer-adaptive handwritten text
recognition." Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition. 2021. [Arxiv link](https://arxiv.org/abs/2104.01876)

[2] Li, Hui, et al. "Show, attend and read: A simple and strong baseline for irregular
text recognition." Proceedings of the AAAI conference on artificial intelligence.
Vol. 33. No. 01. 2019. [Arxiv link](https://arxiv.org/abs/1811.00751)

[3]  Singh, Sumeet S., and Sergey Karayev. "Full page handwriting recognition via image
to sequence extraction." International Conference on Document Analysis and Recognition.
Springer, Cham, 2021. [Arxiv link](https://arxiv.org/abs/2103.06450)
