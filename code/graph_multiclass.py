import matplotlib.pyplot as plt
import numpy as np

binary_validationF = np.matrix([
    [0.63896,   0.56448],
    [0.63849,   0.56658],
    [0.63750,   0.56917],
    [0.63723,   0.56885],
    [0.63723,   0.56885]
   ])

def plot_binary(x, ys, metric, title):
    fig = plt.figure()
    plt.plot(x, ys[:,0], label="No readmission")
    plt.plot(x, ys[:,1], label="Readmission")
    plt.xlabel("Regularization Parameter Lambda")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.savefig(title)
    plt.show()

def plot_multi(x, ys, metric, title):
    fig = plt.figure()
    axes = plt.gca()
    plt.plot(x, ys[:,0], label="No readmission")
    plt.plot(x, ys[:,1], label="Readmission within 30 days")
    plt.plot(x, ys[:,2], label="Readmission after 30 days")
    plt.xlabel("Regularization Parameter Lambda")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.savefig(title)
    plt.show()

def main():
    num_iters = [100, 200, 400, 800, 1600]
    plot_binary(num_iters, binary_validationF, "F1", "Binary Logistic Regression: F1 by Number of Iterations")
    #plot_multi(num_iters, multi_validationF, "F1", "Multiclass Logistic Regression: F1 by Number of Iterations")

if __name__ == "__main__":
    main()