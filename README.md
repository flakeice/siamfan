# SiamFAN: Siamese Feature Attention Network for Ship Tracking in Complex Maritime Environments

## Test tracker

```bash
cd experiments/siamfan
python -u ../../tools/test.py --snapshot model.pth --dataset LMDTSHIP --config config.yaml
```

The testing results will in the current directory (results/dataset/model_name).


## Eval tracker

```
python ../../tools/eval.py --tracker_path ./results --dataset LMDTSHIP --num 1 --tracker_prefix 'model'
```

## Training

See [TRAIN.md](TRAIN.md) for detailed instruction.

