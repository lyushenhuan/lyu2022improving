import numpy as np
import matplotlib.pyplot as plt

# ############MNIST_Fraction
frac = np.array([0.125, 0.25, 0.5, 1, 5, 10, 20, 100])
x = np.log2(frac)
xent = np.array([0.7255, 0.8172, 0.8732, 0.9001, 0.9600, 0.9717, 0.9782, 0.9909])
mlm = np.array([0.7066, 0.7966, 0.8612, 0.8941, 0.9552, 0.9652, 0.9753, 0.9895])
odn = np.array([0.7750, 0.8518, 0.9202, 0.9415, 0.9726, 0.9780, 0.9834, 0.9916])
smlm = np.array([0.7447, 0.8236, 0.9009, 0.9273, 0.9711, 0.9776, 0.9830, 0.9914])

plt.figure(figsize=(5, 6))

ax = plt.subplot(2,1,1)
plt.plot(x, xent, "--", label="xent", color="green", linewidth=1.5)
plt.plot(x, mlm, "--", label="hinge", color="purple", linewidth=1.5)
plt.plot(x, smlm, "--", label="soft hinge", color="blue", linewidth=1.5)
plt.plot(x, odn, "--", label="mdNet", color="red", linewidth=1.5)

group_labels = ['0.125', '0.25', '0.5', '1', '5', '10', '20', '100']
plt.xticks(x, group_labels, rotation=0)
plt.grid(axis='x', linewidth=0.75, linestyle='--', color='0.75')
plt.grid(axis='y', linewidth=0.75, linestyle='--', color='0.75')
plt.xlabel("Fraction of Training Data (%)")
plt.ylabel("Test Accuracy")
plt.title("MNIST")
plt.ylim(0.70, 1)
plt.legend()



############CIFAR10_Fraction
frac = np.array([0.5, 1, 5, 10, 20, 100])
x = np.log2(frac)
xent = np.array([0.2359, 0.3479, 0.5074, 0.5977, 0.6789, 0.8351])
mlm = np.array([0.2342, 0.3630, 0.4871, 0.5836, 0.6590, 0.8215])
odn = np.array([0.3347, 0.4075, 0.5873, 0.6565, 0.7273, 0.8461])
smlm = np.array([0.2080, 0.3572, 0.5139, 0.5944, 0.6821, 0.8196])

ax = plt.subplot(2,1,2)
plt.plot(x, xent, "--", label="xent", color="green", linewidth=1.5)
plt.plot(x, mlm, "--", label="hinge", color="purple", linewidth=1.5)
plt.plot(x, smlm, "--", label="soft hinge", color="blue", linewidth=1.5)
plt.plot(x, odn, "--", label="mdNet", color="red", linewidth=1.5)

group_labels = ['0.5', '1', '5', '10', '20', '100']
plt.xticks(x, group_labels, rotation=0)
plt.grid(axis='x', linewidth=0.75, linestyle='--', color='0.75')
plt.grid(axis='y', linewidth=0.75, linestyle='--', color='0.75')
plt.xlabel("Fraction of Training Data (%)")
plt.ylabel("Test Accuracy")
plt.title("CIFAR10")
plt.ylim(0.20, 0.86)
plt.legend(loc='lower right')
plt.show()
