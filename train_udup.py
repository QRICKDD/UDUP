import os
from UDUP.UDUPAttack import RepeatAdvPatch_Attack
import logging
from AllConfig.GConfig import abspath
import warnings
warnings.filterwarnings("ignore")
if __name__ == '__main__':
    eps = 120
    is_PI = False
    decay = 0.1
    step_alpha = 3
    lm_mui_thre=0.06#[0.05-0.0]
    size_list=[30]
    lambdaw_list=[0.1]

    save_mui=[0.09]
    for size in size_list:
        print("size:",size)
        for lambdaw in lambdaw_list:
            print("lambdaw:{}".format(lambdaw))
            RAT = RepeatAdvPatch_Attack(data_root=os.path.join(abspath, "AllData"),
                                        savedir=os.path.join(abspath,
                                                             'result_save_abslation/size={}_step={}'
                                                             '_eps={}_lambdaw={}'.format(size,step_alpha,
                                                                                                   eps,
                                                                                                   lambdaw)),
                                        log_name=os.path.join(abspath,
                                                              "Mylog_abslation",'size={}_step={}_'
                                                                      'eps={}_lambdaw={}.log'.format(size,step_alpha,
                                                                                                         eps,
                                                                                                         lambdaw)),
                                        alpha=step_alpha / 255, batch_size=100, gap=5, T=120,
                                        lambdaw=lambdaw,
                                        eps=eps / 255, decay=decay,
                                        adv_patch_size=(1, 3, size, size),
                                        model_name="CRAFT",
                                        is_PI=is_PI,save_mui=save_mui,
                                        lm_mui_thre=lm_mui_thre)
            RAT.train()
            del(RAT)
            logging.shutdown()

