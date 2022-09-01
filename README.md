The PyTorch code for paper: 《Knowledge Aware Emotion Recognition in Textual Conversations via Multi-Task Incremental Transformer》

The model is based on [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) and [KET](https://github.com/zhongpeixiang/KET)

## Steps

- Download data: download data to respective folder in `./data/`: `EC`, `DD`, `MELD`, `EmoryNLP`, and `IEMOCAP`.
- Install [Magnitude Medium GloVe](https://github.com/plasticityai/magnitude) for pretrained word embedding.
or first download the `.txt` format word vector file, then convert `.txt` format word vector file to `.magnitude` format word vector file and store to folder in `~/WordEmbedding/`
- Preprocess data: run `preprocess.py` to process `csv` or `pkl` (IEMOCAP) files into `pkl` data.
- Download [ConceptNet 5.6.0](https://github.com/commonsense/conceptnet5/wiki/Downloads) to respective foler in `./data/KB/5.6.1/`.
- Preprocess ConceptNet: run `preprocess_conceptnet.py` 
- Model training: run `train.py`. 
- Model evaluation: run `train.py` with `test_mode` set.
