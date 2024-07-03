import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns


def count(min_value,max_value,interval,intervalNum,margin):
    cur = 0
    value = np.zeros(intervalNum,dtype="int")
    for i in range(intervalNum):
        temp1 = margin<(min_value+(i+1)*interval)
        temp2 = margin>=(min_value+i*interval)
        value[i] = np.sum(temp1*temp2)
    return value


def draw_margin(margin_list, name_list, intervalNum = 15):
    num_bin = len(margin_list)
    print(num_bin)
    margin_all = np.array(margin_list)
    min_value = -0.05
    max_value = 0.61
    interval = (max_value - min_value)/ intervalNum
    x = np.linspace(min_value,max_value,intervalNum,endpoint=False)
    width = interval / (num_bin + 1)
    # color_list=['k','g','b','r','y']
    color_list=['#FF9300','#00A2FF','#88FA4E','#FF644E']

    plt.figure(figsize=(5,2.5))
    for i in range(num_bin):
        value = count(min_value,max_value,interval,intervalNum,margin_list[i])
        plt.bar(x + width*(i+1), value, width = width, label=name_list[i],color = color_list[i])
    plt.xlabel('margin')
    plt.ylabel('number of samples')
    plt.legend()
    plt.show()


    

    # name_list = ['Monday','Tuesday','Friday','Sunday']
    # num_list = [1.5,0.6,7.8,6]
    # num_list1 = [1,2,3,1]
    # x =list(range(len(num_list)))
    # total_width, n = 0.8, 2
    # width = total_width / n

     
    # plt.bar(x, num_list, width=width, label='boy',fc = 'y')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    # plt.bar(x, num_list1, width=width, label='girl',tick_label = name_list,fc = 'r')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    # dataset = "HAR"
    # module = ['cascade']
    # f = open("./result/{}_result.txt".format(dataset),"w")
    # info_list, result = train_and_fit(module,dataset)
    # for info in info_list:
    #   f.write(info+"\n")
    # f.close()
    
    name_list = ['./margin/MNIST_Xent_margin.pkl','./margin/MNIST_MLM_margin.pkl','./margin/MNIST_SMLM_margin.pkl','./margin/MNIST_ODN_margin.pkl']
    l = ['xent','hinge','soft hinge','mdNet']
    margin_list = []
    # for name in name_list:
    #   margin_list.append(torch.load(name).numpy())
    # draw_margin(margin_list, l)

    color_list=['purple','g','b','r']
    plt.figure(figsize=(5,2.5))
    i = 0
    for name in name_list:
        margin = torch.load(name).numpy()
        # sns.distplot(margin,hist=False, kde_kws=dict(cumulative=False),color = color_list[i], label=l[i])
        sns.kdeplot(margin, shade=True, color = color_list[i], label=l[i]) 
        i += 1
    plt.xlim([-0.00,0.5])
    plt.ylim([-0.00,7.5])
    plt.xlabel('margin')
    plt.ylabel('PDF')
    plt.legend()
    plt.show()










    # name_list = []
    # l = []
    # for i in [0,1,2,5]:#range(20):
    #     name_list.append('cascade_{}.txt'.format(i))
    #     l.append('Layer {}'.format(i))
    # margin_list = []
    # for name in name_list:
    #     margin_list.append(np.loadtxt(name))
    # draw_margin(margin_list,l)
    
    