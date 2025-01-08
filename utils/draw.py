import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
from pandas.plotting import table
import pandas
plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


def plot_relative_distance_and_absolute_velocity(dis, vel, path, name, show=True):
    d = dis.shape[1]
    fig, a = plt.subplots(d, 2)
    for i in range(d):
        if d == 1:
            a[0].plot(vel[:, 0], color="r", label='ego')
            a[0].plot(vel[:, i+1], label=f'v{i+1}')
            a[0].set_title('velocity')
            a[0].legend()

            a[1].plot(dis[:, i])
            a[1].set_title('distance between two vehicles')
        else:
            a[i][0].plot(vel[:, 0], color="r", label='ego')
            a[i][0].plot(vel[:, i + 1], label=f'v{i + 1}')
            a[i][0].set_title('velocity')
            a[i][0].legend()

            a[i][1].plot(dis[:, i])
            a[i][1].set_title('distance between two vehicles')

    plt.tight_layout()
    plt.savefig(f"{path}/{name}.png")

    if show:
        plt.show()


def plot_table(data, path, name, rows_name, cols_name=None, save_excel=False, value_name="Log Likelihood"):

    cols_name.append(f"Mean({value_name})")
    data_copy = numpy.array(data)
    data_mean = data_copy.mean(axis=1).reshape(-1, 1)
    data_copy = numpy.concatenate((data_copy, data_mean), axis=1)
    df = pandas.DataFrame(data_copy.T,
                          columns=pandas.Index(rows_name, name="Flow"),
                          index=pandas.Index(cols_name, name="Dataset")).round(4)

    fig = plt.figure(dpi=1400)
    ax = fig.add_subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    table(ax, df, loc="center", cellLoc="center", rowLoc="center", rowLabels="Dataset", colLabels="Flow")
    plt.tight_layout()
    plt.savefig(f"{path}/{name}.png")
    try:
        if save_excel:
            df.to_excel(f"{path}/{name}.xlsx")
    except Exception as e:
        print(e)
        print("please download openpyxl, like: pip install openpyxl")