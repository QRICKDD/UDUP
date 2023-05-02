import subprocess

def runcmd():
    i=1
    while True:

        ret = subprocess.run([r'C:\Users\think\anaconda3\envs\OCR\python.exe',r'F:\OCR-TASK\OCR__advprotect\ATK\RepeatAdvPatchAttack.py'],
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        #print(ret.returncode)
        print("{}-th reboot".format(i))
        i+=1
runcmd()