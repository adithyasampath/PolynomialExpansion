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

The model file used can be found at `models\transformer\nlayers3hdim256\best_model_full_epoch50.pth`. 
1. The `--exp_name` and `--best_epoch` arguments are used to fetch the required model. Here, the `exp_name` is `nlayers3hdim256`, to signify a Transformer enoder-decoder model with 3 layers, 8 attention heads, and hidden dim of 256. 
2. The `--best_epoch` is `50`, which is the epoch number where best validation loss was obtained. 

Also, prints results on 10 equations each for both val and test data.

# Results
```
---- Example 0 ----
src = (o-2)*(3*o+31)
trg = 3*o**2+25*o-62
prd = 3*o**2+25*o-62
score = 1



---- Example 1 ----
src = (5*h-5)*(6*h-28)
trg = 30*h**2-170*h+140
prd = 30*h**2-170*h+140
score = 1



---- Example 2 ----
src = (h+21)*(h+29)
trg = h**2+50*h+609
prd = h**2+50*h+609
score = 1



---- Example 3 ----
src = (-9*z-25)*(7*z-29)
trg = -63*z**2+86*z+725
prd = -63*z**2+86*z+725
score = 1



---- Example 4 ----
src = (-5*n-29)*(-3*n-12)
trg = 15*n**2+147*n+348
prd = 15*n**2+147*n+348
score = 1



---- Example 5 ----
src = (13-2*c)*(-9*c-8)
trg = 18*c**2-101*c-104
prd = 18*c**2-101*c-104
score = 1



---- Example 6 ----
src = (-8*i-29)*(i+19)
trg = -8*i**2-181*i-551
prd = -8*i**2-181*i-551
score = 1



---- Example 7 ----
src = s*(-6*s-20)
trg = -6*s**2-20*s
prd = -6*s**2-20*s
score = 1



---- Example 8 ----
src = (27-t)*(t-25)
trg = -t**2+52*t-675
prd = -t**2+52*t-675
score = 1



---- Example 9 ----
src = -7*o*(-7*o-27)
trg = 49*o**2+189*o
prd = 49*o**2+189*o
score = 1
```