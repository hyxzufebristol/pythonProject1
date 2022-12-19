# -*- coding: utf-8 -*- 
# @Time : 2022/12/19 19:15 
# @Author : YeMeng 
# @File : vis3.py 
# @contact: 876720687@qq.com
# TODO:没有用


# import matplotlib.pyplot as plt
# import numpy as np
# # Fixing random state for reproducibility
# np.random.seed(19680801)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# # 输入x y 值
# x, y = np.random.rand(2, 5)
# # 对应2d z值
# hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 2], [0, 2]])
# # Construct arrays for the anchor positions of the 16 bars.
# xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
# xpos = xpos.ravel()
# ypos = ypos.ravel()
# zpos = 0
# # Construct arrays with the dimensions for the 16 bars.
# dx = dy = 0.5 * np.ones_like(zpos)
# dz = hist.ravel()
# ax.bar3d(xpos, ypos, zpos, dx, dy, dz)
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
colors = ['r', 'g', 'b', 'y']
yticks = [3, 2, 1, 0]
for c, k in zip(colors, yticks):
    # Generate the random data for the y=k 'layer'.
    xs = np.arange(20)
    ys = np.random.rand(20)
    # You can provide either a single color or an array with the same length as
    # xs and ys. To demonstrate this, we color the first bar of each set cyan.
    cs = [c] * len(xs)
    # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
    ax.bar(xs, ys, zs=k, zdir='y', color=cs, alpha=0.8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# On the y-axis let's only label the discrete values that we have data for.
ax.set_yticks(yticks)
plt.show()



# 3d
# df_unstacked = df_grouped3.unstack()
# x = df_unstacked.index.to_list()
# y = df_unstacked.columns.to_list()
# zpos=0
# dz = df_unstacked.values.ravel()
# # Create a figure and an axis
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# # Use ax.bar3d to create a 3D bar chart
# ax.bar3d(x=np.arange(len(x)),
#          y=np.arange(len(y)),
#          z=zpos,
#          dx=0.5,
#          dy=0.5,
#          dz=dz)
# # Set the x- and y-axis labels
# ax.set_xticks(np.arange(len(x)))
# ax.set_yticks(np.arange(len(y)))
# ax.set_xticklabels(x)
# ax.set_yticklabels(y)
# # Show the plot
# plt.show()



