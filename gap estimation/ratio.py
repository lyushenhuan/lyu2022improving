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
#             fig = plt.figure(figsize=(8,5))
#             ax = plt.subplot(1,2,1)
#         if module=="test":
#             ax = plt.subplot(1,2,2)
#         plt.plot(x,xent,label="xent",color="green",linewidth=1.5)
#         plt.plot(x,mlm,label="hinge",color="purple",linewidth=1.5)
#         plt.plot(x,smlm,label="soft hinge",color="blue",linewidth=1.5)
#         plt.plot(x,odn1,label="odn",color="red",linewidth=1.5)
#         plt.plot(x,odn0,label="s-odn",color="black",linewidth=1.5)
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy (%)')
#         plt.xlim([0,300])
#         plt.legend(loc="lower right")
#         ax.set_title("{} accuracy curve".format(module),verticalalignment="bottom")
#         if module=="test":
#             plt.show()

######################################################################################################

dataset = "mnist"
color = ['red','blue','green','purple','black','orange']
fig = plt.figure(figsize=(5,3))
idx = 0
for module in ["train"]:
    for opt in ["SGD"]: #"Adam", "RMSprop", 
    # x = np.loadtxt("./result/{}_xent_{}_epoch_list.txt".format(dataset,opt))
    # xent = np.loadtxt("./result/{}_xent_{}_{}_list.txt".format(dataset,opt))
    # mlm = np.loadtxt("./result/{}_mlm_{}_{}_list.txt".format(dataset,opt))
    # smlm = np.loadtxt("./result/{}_smlm_{}_{}_list.txt".format(dataset,opt))
    # odn1 = np.loadtxt("./result/{}_1_{}_{}_list.txt".format(dataset,opt))
    # odn0 = np.loadtxt("./result/{}_0_{}_{}_list.txt".format(dataset,opt))
        x = np.loadtxt("./result100/{}_0_{}_epoch_list.txt".format(dataset,opt))
        trainacc = np.loadtxt("./result100/{}_0_{}_train_list.txt".format(dataset,opt,module))
        testacc = np.loadtxt("./result100/{}_0_{}_test_list.txt".format(dataset,opt,module))
        trainrate = np.loadtxt("./result100/{}_0_{}_train_rate_list.txt".format(dataset,opt,module))
        testrate = np.loadtxt("./result100/{}_0_{}_test_rate_list.txt".format(dataset,opt,module))

        ax1 = plt.subplot(1,1,1)
        t1 = plt.plot(x[5:55],trainacc[5:55],'--',label="train acc".format(opt),color='red',linewidth=1.5)
        plt.plot(x[5:55],testacc[5:55],label="test acc".format(opt),color='red',linewidth=1.5)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc="upper center", bbox_to_anchor=(0.8,0.85))

        ax2 = ax1.twinx()
        plt.plot(x[5:55],trainrate[5:55],'--',label="train rate".format(opt),color='blue',linewidth=1.5)
        plt.plot(x[5:55],testrate[5:55],label="test rate".format(opt),color='blue',linewidth=1.5)
        plt.ylabel('std/mean')
        plt.legend(loc="upper center", bbox_to_anchor=(0.8,0.35))
        # ax1 = plt.subplot(1,1,1)
        # plt.plot(x,trainacc,label="train acc".format(opt),color='red',linewidth=0.5)
        # plt.plot(x,testacc,label="test acc".format(opt),color='purple',linewidth=0.5)

        # ax2 = ax1.twinx()
        # plt.plot(x,trainrate,label="train rate".format(opt),color='blue',linewidth=0.5)
        # plt.plot(x,trainrate,label="test rate".format(opt),color='green',linewidth=0.5)


        
        
        #ax.set_title("{}".format(module),verticalalignment="bottom")
# ax1.legend(loc='center right') 
# ax2.legend(loc='center right')
plt.show()