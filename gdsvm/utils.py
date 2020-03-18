import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import hashlib
import matplotlib.animation as animation


def color_from_name(name):
    # Make colors from names
    r = ((hash(name) * 1235 + 15) % 256) / 256
    v = ((hash(name) * 66985 + 15) % 256) / 256
    b = ((hash(name) * 32145 + 15) % 256) / 256

    return (r, v, b)


def plot_heatmap(fun, xmin, xmax, ymin, ymax, points_per_axis=100):
    """
    This plot a 2D functions using heat map for the values
    :param fun:
    :param xmin:
    :param xmax:
    :param ymin:
    :param ymax:
    :param points_per_axis:
    :return:
    """
    X = np.linspace(xmin, xmax, num=points_per_axis)
    Y = np.linspace(ymin, ymax, num=points_per_axis)

    heatmap = np.zeros((points_per_axis, points_per_axis))

    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            heatmap[j, i] = fun(np.array([x, y]))

    plt.figure(1, figsize=(10, 10), dpi=100)
    im = plt.imshow(heatmap, cmap='hot', interpolation='nearest', extent=(Y[0], Y[-1], X[0], X[-1]), origin="lower")
    plt.colorbar(im)


def plot_gd(path, color, *args, **kwargs):
    """Draws arrows between each gradient step"""
    init, grads = path

    for g in grads:
        plt.arrow(init[0], init[1], g[0], g[1], color=color, shape="right")
        init += g


def plot_gd2(path, label=None):
    """
    Use graph plotting for animating the video
    :param path:
    :param label:
    :return:
    """
    init, grads = path

    points = [init]

    for g in grads:
        points.append(points[-1] + g)

    points = np.array(points)

    plt.plot(points[:, 0], points[:, 1], label=label)
    plt.legend()


def init_plot(all_path, ax):
    plot_list = []
    for _, _, name in all_path:
        l, = ax.plot([], [], label=name)
        plot_list.append(l)

    return plot_list


def animation_function(i, all_path, plots_list, ax):
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)

    for k, (init, grads, name) in enumerate(all_path):
        points = [init]

        for g in grads[:i]:
            points.append(points[-1] + g)

        points = np.array(points)

        plots_list[k].set_data(points[:, 0], points[:, 1])

    return plots_list


def heatmap_init(fun, xmin, xmax, ymin, ymax, points_per_axis=100):
    X = np.linspace(xmin, xmax, num=points_per_axis)
    Y = np.linspace(ymin, ymax, num=points_per_axis)

    heatmap = np.zeros((points_per_axis, points_per_axis))

    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            heatmap[j, i] = fun(np.array([x, y]))

    im = plt.imshow(heatmap, cmap='hot', interpolation='nearest', extent=(Y[0], Y[-1], X[0], X[-1]), origin="lower")
    im = plt.colorbar(im)
    return im,


def animate(path, all_path, loss, frames=300):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots()

    plots_list = init_plot(all_path, ax)
    plt.legend()

    ani = animation.FuncAnimation(fig, lambda i: animation_function(i, all_path, plots_list, ax),
                                  init_func=lambda: heatmap_init(loss, -20, 20, -20, 20),
                                  frames=frames, interval=1, repeat=False)

    if path:
        ani.save(filename=path, writer=writer)
    else:
        return ani.to_html5_video()
