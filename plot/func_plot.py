import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure(1, (4, 2))
x = np.arange(-0.9, 4, 0.1)
y = np.zeros(len(x))
for i in range(len(x)):
	y[i] = (x[i] <= 0.5)*np.power(x[i]-0.5, 2) + (x[i] > 0.5 and x[i] < 2.5)*0 + (x[i] >= 2.5)*np.power(x[i] - 2.5, 2)


plt.xticks([ 0, 0.5, 2.5], [str(0), r"$r - \theta$", r"$r + \theta$"])
plt.yticks([0, 0.5, 1, 1.5, 2])
plt.grid(linewidth=0.75, linestyle='--', color='0.5')
# plt.title("mdNet Loss Function")
plt.plot(x, y, color="red", linewidth=2)
plt.show()

# import matplotlib.pyplot as plt
# from mpl_toolkits.axisartist.axislines import SubplotZero
# import numpy as np

# fig = plt.figure(1, (8, 5))

# # a subplot with two additional axis, "xzero" and "yzero". "xzero" is
# # y=0 line, and "yzero" is x=0 line.
# ax = SubplotZero(fig, 1, 1, 1)
# fig.add_subplot(ax)

# # make xzero axis (horizontal axis line through y=0) visible.
# ax.axis["xzero"].set_visible(True)
# ax.axis["xzero"].label.set_text("Margin")

# # make other axis (bottom, top, right) invisible.
# for n in ["bottom", "top", "right"]:
#     ax.axis[n].set_visible(False)

# x = np.arange(-1, 4, 0.1)
# y = np.zeros(len(x))
# for i in range(len(x)):
# 	y[i] = (x[i] <= 0.5)*np.abs(x[i]-0.5) + (x[i] > 0.5 and x[i] < 2.5)*0 + (x[i] >= 2.5)*np.power(x[i] - 2.5, 2)


# # plt.xticks([-1, 0, 0.5, 1, 2, 2.5, 3, 4], [str(-1), str(0), r"$r - \theta$", str(1), str(2), r"$r + \theta$", str(3), str(4)])
# plt.xticks([ 0, 0.5, 2.5], [str(0), r"$r - \theta$", r"$r + \theta$"])
# plt.yticks([-0.5, 0, 0.5, 1, 1.5, 2])
# plt.grid(color="k", linestyle=":")
# plt.title("ODN Loss Function")
# ax.plot(x, y, color="red", linewidth="3")

# plt.show()