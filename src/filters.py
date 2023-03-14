import numpy as np

def filter_sigmoid(x, a, b, c):
    y = 1 / (1 + a * np.exp(-b * (x - c)))
    return y


if (__name__ == "__main__"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.grid()
    x = np.linspace(-3., 3., 1000)
    a = 1
    b = np.linspace(1, 30, 10)
    c = 0
    for i, bi in enumerate(b):
        y = filter_sigmoid(x, a, bi, c)
        ax.plot(x, y, label=i)
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.grid()
    x = np.linspace(-5., 10., 1000)
    a = 1
    b = 10#50 np.linspace(0.5, 50, 5)
    c = [2.1]
    for i, ci in enumerate(c):
        y = filter_sigmoid(x, a, b, ci)
        ax.plot(x, y, label=i)
    plt.legend()
    plt.show()

