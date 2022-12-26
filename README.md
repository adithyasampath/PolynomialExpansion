# PolynomialExpansion
NLP Project using Seq2seq transformer model for expanding polynomial expressions

# Setup

## Install Pytorch

```pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116```

## Install other requirements
Run the command `pip install -r requirements.txt`

# Train

Run `python train.py --help` to understand all the arguments and hyperparaters.

To train a Transformer enoder-decoder model with 3 layers and 8 attention heads for 50 epochs, run `python train.py` 

# Test

To test the above Transformer enoder-decoder model on `test.txt` and `val.txt`, run `python train.py --test`

Hence, we get val and test accuracy. Saves the metrics in a JSON file.

The model file used can be found at `models\transformer\nlayers3hdim256\best_model_full_epoch50.pth`. The `--exp_name` and `--best_epoch` arguments are used to fetch the required model. Here, the `exp_name` is `nlayers3hdim256`, to signify a Transformer enoder-decoder model with 3 layers, 8 attention heads, and hidden dim of 256. The `--best_epoch` is `50`, which is the epoch number where best validation loss was obtained. 

Also, prints results on 10 equations each for both val and test data.