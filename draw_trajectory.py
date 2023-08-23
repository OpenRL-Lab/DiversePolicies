from tqdm import tqdm
import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
np.random.seed(5)
QUALITY = 500
EPISODES = 55
MIDDLE = np.array([0.5, np.sqrt(3) / 6])
INIT_DIS = 0.1
DOT_NUM = 3
WALK_LEN = 0.1
COLORS = ['red','blue','green','yellow','brown']
A = np.array([
    [1.,1.,1.],
    [0.,1.,.5],
    [0.,0.,np.sqrt(3)/2]
])
intial_points = np.array([
    [0, 1],
    [-np.sqrt(3) / 2, -1 / 2],
    [np.sqrt(3) / 2, -1 / 2]
])

dots = intial_points * 0.06 + (np.random.rand(DOT_NUM, 2) - 0.5) * 0.03 + MIDDLE
dotsize = 60
for episode in tqdm(range(EPISODES)):
    middle_dot = np.mean(dots, 0)
    distance = []
    for dot in dots:
        distance.append(((dot[0] - middle_dot[0]) ** 2 + (dot[1] - middle_dot[1]) ** 2) ** 0.5 / 0.1)
    temp = EPISODES / 10
    WIDTH = 4
    ALPHA = 0.8
    pick_dot = np.random.randint(DOT_NUM)
    if pick_dot == 0:
        plt.scatter(dots[0][0], dots[0][1], c = 'w',edgecolors=[[1, 1 - distance[0] / temp, 0]], linewidth = WIDTH, alpha = ALPHA, s=dotsize)
    elif pick_dot == 1:
        plt.scatter(dots[1][0], dots[1][1], c = 'w',edgecolors=[[0.9 - distance[1] / temp, 0.1, 0.1]], linewidth = WIDTH, alpha = ALPHA, s=dotsize)
    else:
        plt.scatter(dots[2][0], dots[2][1], c = 'w',edgecolors=[[0.2, 0.9 - distance[2] / temp, 0.2]], linewidth = WIDTH, alpha = ALPHA, s=dotsize)
    plt.scatter(middle_dot[0], middle_dot[1], c = 'w',edgecolors=[[0.5 - episode / (EPISODES + 60), 0.9 - episode / (EPISODES + 30), 0.9]], alpha = ALPHA, linewidth = WIDTH, s = 15)
    print(distance[0])
    dots[pick_dot] += (dots[pick_dot] - middle_dot) * WALK_LEN
plt.xticks([]),plt.yticks([])

plt.axis('off')
triangles = tri.Triangulation(A[1], A[2])
plt.triplot(triangles,'-')
plt.savefig('trajectories.png', bbox_inches='tight')