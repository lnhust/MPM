# MPM

## prepare for dataset
download miniImageNet dataset from [here](https://mega.nz/#!rx0wGQyS!96sFlAr6yyv-9QQPCm5OBFbOm4XSD0t-HlmGaT5GaiE) <br>
change the path of dataset in dataloader.py

## miniImageNet 1-shot scenario: <br>
python test.py

## miniImageNet 5-shot scenario: <br>
python test.py --model-path=output/ConvNet/miniImageNet_test_5shot.model --num-cls=5 --num-inst=5 --exp-dir=output/ConvNet/miniImageNet-5shot
