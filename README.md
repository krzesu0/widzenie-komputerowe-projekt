# Computer vision course project repository
## How to set it up?
I didn't properly manage all the python dependencies, but for sure you will need:
```
computer
python
numpy
matplotlib
tensorflow      (with GPU support if you don't want to witness heat death of universe)
pandas          (for visualizations, python's built in csv reader sucks)
Pillow          (python package, not irl, but it also will come in handy whilst waiting)
pygame          (if you want to see how it does with your digits)
python-mnist
mnist image binaries
```

## How to get MNIST files?
Place `*-ubyte` decompressed files into `mnist` folder in the root of repo.


You can get these files here: [click](http://yann.lecun.com/exdb/mnist/).
To unzip use `gunzip <name>` or GUI like 7zip or WinRAR.

## How to get the model checkpoints?
I don't want to spam github with over 2GBs of raw model weights, if you really want them, open a issue and I'll contact you.
If you want to train those models yourself, just type `python train_<name>.py` into terminal and sit back or use the *highly* sophisticated script `train.sh` and go to sleep. 

With my GTX960 I got about 30 secs per epoch, with 2070S, about half of that. `30s * 150 epochs * 5 models ~= 6hrs`

## What is this `interactive.py`?
I've made a small little teensy-weensy sketch, that you can use to see how's your model doing trying to guess your hand-writing.

Start it with: `python interactive.py <path_to_model_file>`, eg. `python interactive.py output/own/model/150`.
For now (and probably ever) it only supports the "own" model due to special preprocessing of mobile and efficient networks inputs.

Use keys `~` through `0` (above keys, not on the keypad) to pick a response layer (`~` is used for the background, and rest of the keys correspond to the class), and keys `q` through `p` to set the brush size. `Backspace` clears the draw area, `Return` predicts, `x` inserts single generated image the same as for training into drawing area and `Escape` quits.

Don't be the cause of AI uprising and stay safe.