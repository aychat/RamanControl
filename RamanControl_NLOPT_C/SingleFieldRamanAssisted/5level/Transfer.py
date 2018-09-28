import numpy as np
import matplotlib.pyplot as plt

N = 60

pop_iter = np.zeros((2, N))

x1 = .87
y1 = .64/.87
z1 = .000
x2 = .46
y2 = .07/.46
z2 = .000

br1 = .85


def transfer(pop, x1, y1, z1, x2, y2, z2, br1, axes, clr1, clr2):

    br2 = 1 - br1
    A = np.array([[1 - (z1 + x1*y1)*br2, (x2*y2 + z2)*br1], [(x1*y1 + z1)*br2, 1 - (z2 + x2*y2)*br1]])

    for i in range(N):
        pop_iter[0, i] = pop[0]
        pop_iter[1, i] = pop[1]

        pop = A.dot(pop)

    axes.plot(pop_iter[0], clr1)
    axes.plot(pop_iter[1], clr2)

    print(pop)


def transfer_direct(pop, x1, x2, br1, axes, clr1, clr2):

    br2 = 1 - br1

    A = np.array([[1 - x1*br2, x2*br1], [x1*br2, 1 - x2*br1]])

    for i in range(N):
        pop_iter[0, i] = pop[0]
        pop_iter[1, i] = pop[1]

        pop = A.dot(pop)

    axes.plot(pop_iter[0], clr1, linewidth=2.)
    axes.plot(pop_iter[1], clr2, linewidth=2.)

    print(pop)


pop = np.array([1., 0.])
fig, axes = plt.subplots(nrows=1, ncols=1)
transfer(pop, .4, .288/.4, .000156, .03636, .02/.03636, .000166, .7, axes, 'r', 'b')
transfer(pop, .9, .9, .00015, .1, .5, .00016, .7, axes, 'r*-', 'b*-')
pop = np.array([1., 0.])
transfer_direct(pop, .748, .041, .7, axes, 'r--', 'b--')

axes.set_xlabel("Number of iterations")
plt.show()