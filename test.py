import numpy as np
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import mean_squared_error as MSE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import matplotlib as mpl
def get_ssim_psnr_mse(structure_name):
    
    origin_result = np.load("./data/movie_03_test_pic_1999_VISp_800.npy")
    result = np.load("./paper_results/movie_03_iter_1999_" + structure_name + "_800_400.npy")
    for i in range(5):
        pic1 = origin_result[i]
        pic1 = pic1.reshape((256,256,1))
        pic2 = result[i]
        plt.imshow(pic1,cmap="gray")
        #plt.show()
        plt.savefig("./figs/" +"origin_"+ structure_name + "_" + str(i) + ".jpg")
        plt.imshow(pic2,cmap="gray")
        #plt.show()
        plt.savefig("./figs/" +"result_"+ structure_name + "_" + str(i) + ".jpg")
    ssim = 0
    psnr = 0
    mse = 0
    ssim_list = []
    psnr_list = []
    mse_list = []
    for i in range(5,400):
        pic1 = origin_result[i]
        pic1 = pic1.reshape((256,256,1))
        if structure_name == "CA_delay":
            pic2 = result[i-5]
        else :
            pic2 = result[i]
        ssim_1 = SSIM(pic1, pic2, multichannel = True)
        psnr_1 = PSNR(pic1, pic2)
        mse_1 = MSE(pic1, pic2)
        ssim += ssim_1
        psnr += psnr_1
        mse += mse_1
        ssim_list.append(ssim_1)
        psnr_list.append(psnr_1)
        mse_list.append(mse_1)
    ssim /= 395
    psnr /= 395
    mse /= 395
    print(structure_name +" : ", ssim, psnr, mse)
    print(structure_name +"_median : ", np.median(ssim_list), np.median(psnr), np.median(mse))
    return ssim, psnr, mse, ssim_list, psnr_list, mse_list
#structure_name_list = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]
structure_name_list = ["VISp"]
structure_name_list_inpaper = structure_name_list
ssim_list_list = []
psnr_list_list = []
for structure_name in structure_name_list :
    ssim, psnr, mse, ssim_list, psnr_list, mse_list= get_ssim_psnr_mse(structure_name)
    ssim_list_list.append(ssim_list)
    psnr_list_list.append(psnr_list)

data = {"structure" : structure_name_list,
        "psnr" : psnr_list_list}
df = DataFrame(data)
plt.figure(figsize = (25,10))
ax = sns.violinplot(data = psnr_list_list)
ax.set_xticklabels(structure_name_list, fontsize = 46)
ax.tick_params(labelsize = 46)
ax.set_xlabel("Brain Area",font={ 'size':46})
ax.set_ylabel("PSNR",font={ 'size':46})
#plt.plot([0,6],[13.95131506967481   ,13.95131506967481   ],color="black",linestyle = "-.")
plt.savefig("./figs/psnr.eps",dpi=1000)
