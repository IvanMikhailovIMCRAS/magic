import matplotlib.pyplot as plt
import numpy as np
from outputs import show_exp_var
from scipy.signal import savgol_filter
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


if __name__ == '__main__':
    # reading spectra data
    X = np.loadtxt("Xmm.txt")
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