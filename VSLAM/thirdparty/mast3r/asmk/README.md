# Python ASMK (Aggregated Selective Match Kernels)

This is a Python implementation of the ASMK approach published in [ICCV 2013](http://hal.inria.fr/docs/00/86/46/84/PDF/iccv13_tolias.pdf):

```
@InProceedings{TAJ13,
  author       = "Giorgos Tolias and Yannis Avrithis and Herv\'e J\'egou",
  title        = "To aggregate or not to aggregate: Selective match kernels for image search",
  booktitle    = "IEEE International Conference on Computer Vision",
  year         = "2013"
}
```

This package is provided to support image retrieval with local descriptors and to reproduce the results of our [ECCV 2020 paper](https://arxiv.org/abs/2007.13172) with HOW deep local descriptors:

```
@InProceedings{TJ20,
  author      = "Giorgos Tolias and Tomas Jenicek and Ond\v{r}ej Chum}",
  title       = "Learning and aggregating deep local descriptors for instance-level recognition",
  booktitle   = "European Conference on Computer Vision",
  year        = "2020"
}
```

There are minor differences compared to the original ASMK approach (ICCV'13) and [implementation](https://github.com/gtolias/asmk), which are described in our ECCV'20 paper. Using the provided package to run ASMK with other local descriptors is straightforward.



## Running the Code

1. Install the requirements (`faiss-cpu` for cpu-only setup)

```
pip3 install pyaml numpy faiss-gpu
```


2. Build C library for your Python version

```
python3 setup.py build_ext --inplace
rm -r build
```


3. Download `cirtorch` and add it to your `PYTHONPATH`

```
wget "https://github.com/filipradenovic/cnnimageretrieval-pytorch/archive/master.zip"
unzip master.zip
rm master.zip
export PYTHONPATH=${PYTHONPATH}:$(realpath cnnimageretrieval-pytorch-master)
```


4. Run `examples/demo_how.py` giving it any `.yaml` parameter file from `examples/params/*.yml`


### Reproducing ECCV 2020 results with HOW local descriptors

Reproducing results from **Table 2.**

- R18<sub>how</sub> (n = 1000): &nbsp; `examples/demo_how.py eccv20_how_r18_1000` &ensp; _ROxf (M): 75.1, RPar (M): 79.4_
- -R50<sub>how</sub> (n = 1000): &nbsp; `examples/demo_how.py eccv20_how_r50-_1000` &ensp; _ROxf (M): 78.3, RPar (M): 80.1_
- -R50<sub>how</sub> (n = 2000): &nbsp; `examples/demo_how.py eccv20_how_r50-_2000` &ensp; _ROxf (M): 79.4, RPar (M): 81.6_
