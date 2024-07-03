import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np

# module = "test"
# dic = "./result/LMDF_HAR_{}_prob_layer".format(module)
# color = np.loadtxt("./HAR/label_{}.txt".format(module))
# for t in range(20):
#   path = dic + "{}.txt".format(t)
#   X = np.loadtxt(path)
#   print(X.shape)
#   plt.title('Layer {} on {}'.format(t,module))
#   tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=100)
#   Y = tsne.fit_transform(X)
#   np.savetxt("{}_{}.txt".format(module,t),Y)
#   Y = np.loadtxt("{}_{}.txt".format(module,t))
#   plt.scatter(Y[:, 0], Y[:, 1], s=5, c=color, cmap=plt.cm.Spectral)
#   plt.savefig("{}_{}.pdf".format(module,t))
#   plt.show()
#   print("finish {}\n".format(t))


# module = "train"
# dic = "./result/LMDF_HAR_{}_prob_layer".format(module)
# color = np.loadtxt("../dataset/HAR/label_{}.txt".format(module))

dataset_list = ['CIFAR10','MNIST']
for dataset in dataset_list:

    module_list = ['train','test']
    for module in module_list:
        color = np.loadtxt("./embed/{}_label_{}.txt".format(dataset,module))

        fig = plt.figure(figsize=(14,3))
        ax = plt.subplot(1,4,1)
        Y = np.loadtxt("./embed/{}_Xent_{}.txt".format(dataset,module))
        plt.scatter(Y[:, 0], Y[:, 1], s=5, c=color, cmap=plt.cm.Spectral)
        # plt.yticks([-10,0,10])
        # plt.xticks([-10,0,10])
        ax.set_title("Cross-entropy Loss",verticalalignment="bottom")

        ax = plt.subplot(1,4,2)
        Y = np.loadtxt("./embed/{}_MLM_{}.txt".format(dataset,module))
        plt.scatter(Y[:, 0], Y[:, 1], s=5, c=color, cmap=plt.cm.Spectral)
        # plt.yticks([-10,0,10])
        # plt.xticks([-10,0,10])
        ax.set_title("Hinge Loss",verticalalignment="bottom")

        ax = plt.subplot(1,4,3)
        Y = np.loadtxt("./embed/{}_SMLM_{}.txt".format(dataset,module))
        plt.scatter(Y[:, 0], Y[:, 1], s=5, c=color, cmap=plt.cm.Spectral)
        # plt.yticks([-10,0,10])
        # plt.xticks([-10,0,10])
        ax.set_title("Soft Hinge Loss",verticalalignment="bottom")

        ax = plt.subplot(1,4,4)
        Y = np.loadtxt("./embed/{}_ODN_{}.txt".format(dataset,module))
        plt.scatter(Y[:, 0], Y[:, 1], s=5, c=color, cmap=plt.cm.Spectral)
        # plt.yticks([-10,0,10])
        # plt.xticks([-10,0,10])
        ax.set_title("mdNet Loss",verticalalignment="bottom")

        plt.savefig("./pdf/{}_{}.pdf".format(dataset,module))
        plt.show()