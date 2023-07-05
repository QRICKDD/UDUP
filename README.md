# UDUP
Code for Universal Defensive Underpainting Patch: Making Your Text Invisible to Optical Character Recognition



# About Prefiles
All models and data can be accessed in out code (../AllData/ and ../AllConfig/all_model).
In particular, the history of my running records can be viewed in Mylog.

After downloading, please put them in the prefile folder.

# Dependencies
The code for our paper runs with Python 3.8. Please pip install the following packages:
* requirements.yml


# Running in Windows, MacOS or Ubuntu
We provide as an example the source code to run. Run the following commands in shell:

```shell
###############################################
# Omit if already git cloned.
git clone repository
# Carry out UDUP.
python run train_udup.py
############################################### 
# Results are stored in Mylog and result_save.
# Ground-truth boxes about CRAFT/DBnet/easyocr are stored in test_save
```

# Performance on EAST, Google Tessarct, Oneplus9 and PaddlePaddle
![avatar](https://github.com/QRICKDD/UDUP/blob/main/DEMO-EAST-Tessarct-OnePlus9-PaddlePaddle.png)


# Performance on CRAFT after BM3D denoise
![avatar](https://github.com/QRICKDD/UDUP/blob/main/BM3D-Denoise-Demo.png)
