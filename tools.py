import numpy as np
import settings
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
from matplotlib.pyplot import MultipleLocator
def build_G_from_S(S, k):
    # S: similarity matrix
    # k: number of nearest neighbors
    # G: graph
    G = torch.ones(S.shape).cuda() * -1.5
    # G = torch.zeros(S.shape).cuda()
    G_ = torch.where(S > settings.threshold, S, -1.5).cuda()
    # G_ = S_
    # S_ = torch.where(S > 0 and S > settings.rthreshold, 1, S).cuda()
    # S_ = torch.where(S < 0 and S < settings.lthreshold, -1, S_).cuda()
    # G_= torch.where(S_ != 0, S, 0).cuda()
    for i in range(G_.shape[0]):
        idx = torch.argsort(-G_[i])[:k]
        G[i][idx] = G_[i][idx]
    del G_
    torch.cuda.empty_cache()
    return G

def generate_robust_S(s, alpha=2, beta=2):
    # G: graph
    # S: similarity matrix
    # alpha: parameter for postive robustness
    # beta: parameter for negative robustness
    # S_robust: robust similarity matrix

    S = s

    # find maximum count of cosine distance

    max_count = 0
    max_cos = 0

    interval = 1/1000
    cur = -1.0
    for i in range(2000):
        cur_cnt = np.sum((S>cur) & (S<cur+interval))
        if max_count < cur_cnt:
            max_count = cur_cnt
            max_cos = cur
        cur += interval

    # split positive and negative similarity matrix

    flat_S = S.reshape((-1,1))
    left = flat_S[np.where(flat_S <= max_cos)[0]]
    right = flat_S[np.where(flat_S >= max_cos)[0]]

    # reconstruct
    left = np.concatenate([left, 2 * max_cos - left])
    right = np.concatenate([max_cos-np.maximum(right-max_cos,max_cos-right), right])

    # fit to gaussian distribution
    left_mean, left_std = norm.fit(left)
    right_mean, right_std = norm.fit(right)

    x_major_locator=MultipleLocator(0.5)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    

    # draw the histogram 
    plt.hist(left, bins=10000, density=True, alpha=0.6, color='g')
    plt.savefig('left.png')
    plt.close()
    x_major_locator=MultipleLocator(0.5)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.hist(right, bins=10000, density=True, alpha=0.6, color='r')
    plt.savefig('right.png')
    plt.close()
    x_major_locator=MultipleLocator(0.5)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.hist(flat_S, bins=10000, density=True, alpha=0.6, color='b')
    plt.savefig('flat_S.png')
    plt.close()
    


    print('left mean: ', left_mean)
    print('left std: ', left_std)
    print('threshold:',left_mean-3*left_std)
    print('right mean: ', right_mean)
    print('right std: ', right_std)
    print('threshold:',right_mean+3*right_std)
    # generate robust similarity matrix

    # show the fitting result
    # plt.hist(left, bins=100, density=True, alpha=0.6, color='g')
    # plt.hist(right, bins=100, density=True, alpha=0.6, color='r')
    # plt.show()
    # plt.close()





