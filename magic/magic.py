import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from sklearn import preprocessing
from sklearn.decomposition import PCA


def show_exp_var(exp_var, output_file=''):
    cum_sum_eigenvalues = np.cumsum(exp_var)
    plt.bar(range(0,len(exp_var)), exp_var, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    if output_file == '':
        plt.show()
    else:
        plt.savefig(fname=output_file, dpi=600)
    plt.close()

if __name__ == '__main__':
    # reading spectra data
    X = np.loadtxt("Xmm.txt")
    print(X.shape)
    wavenumbers = np.linspace(1800, 900, X.shape[1])
    with open("mm_labels.txt", "r") as file:
        labels = file.read().split('\n')
    labels.pop(-1)
    lab = ''
    plt.figure(figsize=(20,16))
    cm = plt.cm.get_cmap('tab20')
    counter = -1
    for label, data in zip(labels, X):
        if lab != label:
            lab = label
            counter += 1
            plt.plot(wavenumbers, data, label=lab, color = cm.colors[counter], linewidth=0.5)
        else:
            plt.plot(wavenumbers, data, color = cm.colors[counter], linewidth=0.5)
    plt.xlabel(r"wavenumber, $cm^{-1}$", fontsize=16)
    plt.ylabel(r"ATR units", fontsize=16)
    plt.xlim(1800, 600)
    plt.legend(loc='center right', bbox_to_anchor=(1.0, 0.5), fontsize=12)
    # plt.savefig("Origin.jpg")
    plt.show()
    plt.close()
    # get derivatives
    X = savgol_filter(X, 17, polyorder=2, deriv=2)
    # standardization
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    # PCA processing
    n_pc = 20
    pca = PCA(n_components=n_pc)
    pca.fit(X)
    X = pca.transform(X)
    # show explaned variance curves
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    print('Number of pca components: ', n_pc)
    print('Max cumulative explained variance: ', cum_sum_eigenvalues[-1])
    show_exp_var(exp_var_pca, output_file='explained_variance.jpg')

    from sklearn.manifold import TSNE
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=20).fit_transform(X)
    lab = ''
    counter = -1
    plt.figure(figsize=(20,16))
    for i, label in enumerate(labels):
        if lab != label:
            lab = label
            counter += 1
            plt.plot(X_embedded[i,0], X_embedded[i,1], 'o', label=lab, color = cm.colors[counter])
        else:
            plt.plot(X_embedded[i,0], X_embedded[i,1], 'o', color = cm.colors[counter])
    plt.xlim(-50, 100)
    plt.legend(loc='center right', bbox_to_anchor=(1.0, 0.5), fontsize=13)
    plt.show()
    plt.close()