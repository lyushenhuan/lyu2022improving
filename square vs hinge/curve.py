import numpy as np
import matplotlib.pyplot as plt


# dataset = "mnist"
# for opt in ["SGD", "Adam", "RMSprop"]:
#     for module in ["train","test"]:

#     # x = np.loadtxt("./result/{}_xent_{}_epoch_list.txt".format(dataset,opt))
#     # xent = np.loadtxt("./result/{}_xent_{}_{}_list.txt".format(dataset,opt))
#     # mlm = np.loadtxt("./result/{}_mlm_{}_{}_list.txt".format(dataset,opt))
#     # smlm = np.loadtxt("./result/{}_smlm_{}_{}_list.txt".format(dataset,opt))
#     # odn1 = np.loadtxt("./result/{}_1_{}_{}_list.txt".format(dataset,opt))
#     # odn0 = np.loadtxt("./result/{}_0_{}_{}_list.txt".format(dataset,opt))

#         x = np.loadtxt("./result15/{}_xent_{}_epoch_list.txt".format(dataset,opt))
#         xent = np.loadtxt("./result15/{}_xent_{}_{}_list.txt".format(dataset,opt,module))
#         mlm = np.loadtxt("./result15/{}_mlm_{}_{}_list.txt".format(dataset,opt,module))
#         smlm = np.loadtxt("./result15/{}_smlm_{}_{}_list.txt".format(dataset,opt,module))
#         odn1 = np.loadtxt("./result15/{}_1_{}_{}_list.txt".format(dataset,opt,module))
#         odn0 = np.loadtxt("./result15/{}_0_{}_{}_list.txt".format(dataset,opt,module))

        
#         if module=="train":
#             fig = plt.figure(figsize=(8,4))
#             ax = plt.subplot(1,2,1)
#         if module=="test":
#             ax = plt.subplot(1,2,2)
#         plt.plot(x,xent,label="xent",color="green",linewidth=1.5)
#         plt.plot(x,mlm,label="hinge",color="purple",linewidth=1.5)
#         plt.plot(x,smlm,label="soft hinge",color="blue",linewidth=1.5)
#         plt.plot(x,odn1,label="h-mdNet",color="red",linewidth=1.5)
#         plt.plot(x,odn0,label="s-mdNet",color="black",linewidth=1.5)
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy (%)')
#         plt.xlim([0,20])
#         plt.legend(loc="lower right")
#         ax.set_title("{} accuracy curve".format(module),verticalalignment="bottom")
#         if module=="test":
#             plt.show()

######################################################################################################

dataset = "mnist"
color = ['red','blue','green','purple','black','orange']
fig = plt.figure(figsize=(9,3))
idx = 0
for module in ["train","test"]:
    for opt in ["SGD", "Adam", "RMSprop"]:
    # x = np.loadtxt("./result/{}_xent_{}_epoch_list.txt".format(dataset,opt))
    # xent = np.loadtxt("./result/{}_xent_{}_{}_list.txt".format(dataset,opt))
    # mlm = np.loadtxt("./result/{}_mlm_{}_{}_list.txt".format(dataset,opt))
    # smlm = np.loadtxt("./result/{}_smlm_{}_{}_list.txt".format(dataset,opt))
    # odn1 = np.loadtxt("./result/{}_1_{}_{}_list.txt".format(dataset,opt))
    # odn0 = np.loadtxt("./result/{}_0_{}_{}_list.txt".format(dataset,opt))
        x = np.loadtxt("./result15/{}_xent_{}_epoch_list.txt".format(dataset,opt))
        odn1 = np.loadtxt("./result15/{}_1_{}_{}_list.txt".format(dataset,opt,module))
        odn0 = np.loadtxt("./result15/{}_0_{}_{}_list.txt".format(dataset,opt,module))
        
        if module=="train":
            ax = plt.subplot(1,2,1)
        if module=="test":
            ax = plt.subplot(1,2,2)
        plt.plot(x,odn1,label="h-mdNet_{}".format(opt),color=color[idx],linewidth=1.5)
        plt.plot(x,odn0,label="s-mdNet_{}".format(opt),color=color[idx+1],linewidth=1.5)
        idx += 2
        idx %= 6
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.xlim([0,200])
        # plt.legend(loc="lower right")
        ax.set_title("{}".format(module),verticalalignment="bottom")
ax.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.) 
plt.show()