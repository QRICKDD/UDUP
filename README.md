# UDUP
Code for Universal Defensive Underpainting Patch: Making Your Text Invisible to Optical Character Recognition

![avatar](https://raw.githubusercontent.com/QRICKDD/LLA/master/picture/temp_picture.png)

# About Prefiles
All models and data can be accessed in out code (../AllData/ and ../AllConfig/all_model).
In particular, the running records of experiments can be viewed in Mylog.

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
cd UAUP
# Carry out UDUP.
python run UDUPAttack.py
############################################### 
# Results are stored in Mylog and result_save.
# Ground-truth boxes about CRAFT/DBnet/easyocr are stored in test_save
```


