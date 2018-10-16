import GPy
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
from util import plot_gp
from scipy.stats import multivariate_normal
seed(1234)

# Create data
# lambda function, call f(x) to generate data
f = lambda x: -2*np.cos(2*np.pi*x) + np.sin(6*np.pi*x)
# 10 equally spaced sample locations
X = np.linspace(0.05, 0.95, 10)[:,None]
# y = f(X) + epsilon
Y = f(X) + np.random.normal(0., 0.1, (10,1)) # note that np.random.normal takes mean and s.d. (not variance), 0.1^2 = 0.01


k = GPy.kern.RBF(1, variance=1., lengthscale=0.1, name="rbf")

# New test points to sample function from
num_test_points = 200
Xnew = np.linspace(-0.05, 1.05, num_test_points)[:, None]

# Covariance between training sample points (+ Gaussian noise)
Kxx = k.K(X,X) + 1 * np.eye(10)

# Covariance between training and test points
Kxs = k.K(Xnew, X)

# Covariance between test points
Kss = k.K(Xnew,Xnew)

# The mean of the GP fit (note that @ is matrix multiplcation: A @ B is equivalent to np.matmul(A,B))
mean = Kxs @ np.linalg.inv(Kxx) @ Y
# The covariance matrix of the GP fit
Cov = Kss - Kxs @ np.linalg.inv(Kxx) @ Kxs.T + np.eye(num_test_points) * 0.001
posterior = multivariate_normal(mean=np.squeeze(mean), cov=Cov)


def squashing(X):
    return np.reciprocal(1+np.exp(-X))


Xnew = np.squeeze(Xnew)
mean = np.squeeze(mean)

f, axarr = plt.subplots(2, 2)
"""
First plot some samples of the GP
"""
num_samples = 5
samples = posterior.rvs(size=num_samples)
for i, sample in enumerate(samples):
    axarr[0, 0].plot(Xnew, sample)
axarr[0, 0].set_title('Samples from a posterior GP')
axarr[0, 0].set_ylim([-3, 3])
plot_gp(Xnew, mean, Cov, fig=axarr[0, 0])

"""
# Then plot squashed samples with squashed confidence bounds
"""

for i, sample in enumerate(squashing(samples)):
    axarr[0, 1].plot(Xnew, sample)
axarr[0, 1].set_title('Samples from a posterior GP')
axarr[0, 1].set_ylim([-0.5, 1.5])

mean_squashed = squashing(mean)
upper_squashed = squashing(mean - 1.96 * np.sqrt(np.diag(Cov)))
lower_squashed = squashing(mean + 1.96 * np.sqrt(np.diag(Cov)))

axarr[0, 1].fill_between(Xnew, lower_squashed, upper_squashed, alpha=0.5)
axarr[0, 1].plot(Xnew, mean_squashed, "-", label="Squashed mean function")
axarr[0, 1].plot([0.0, 1.0], [1.0-1E-12, 1.0+1E-12], "r--")
axarr[0, 1].plot([0.0, 1.0], [-1E-12, 1E-12], "r--")
axarr[0, 1].set_title('Squashed samples from a posterior GP with squashed confidence intervals')
axarr[0, 1].legend()

copies = (np.copy(mean_squashed),
          np.copy(upper_squashed),
          np.copy(lower_squashed))

"""
# Then plot squashed samples with estimated confidence bounds
"""
# Now we first squash the functions and then calc the mean and variance
large_sample = squashing(posterior.rvs(size=10000))
std_squashed = np.std(large_sample, axis=0)

mean_squashed = np.mean(large_sample, axis=0)
upper_squashed = mean_squashed - 1.96 * std_squashed
lower_squashed = mean_squashed + 1.96 * std_squashed

for i, sample in enumerate(squashing(samples)):
    axarr[1, 0].plot(Xnew, sample)
axarr[1, 0].set_title('Samples from a posterior GP')
axarr[1, 0].set_ylim([-0.5, 1.5])

axarr[1, 0].fill_between(Xnew, lower_squashed, upper_squashed, alpha=0.5)
axarr[1, 0].plot(Xnew, mean_squashed, "-", label="Squashed mean function")
axarr[1, 0].plot([0.0, 1.0], [1.0-1E-12, 1.0+1E-12], "r--")
axarr[1, 0].plot([0.0, 1.0], [-1E-12, 1E-12], "r--")
axarr[1, 0].set_title('Squashed samples from a posterior GP with estimated confidence intervals')
axarr[1, 0].legend()

"""
Finally, plot the two options side to side in order to compare
"""
axarr[1, 1].plot(Xnew, mean_squashed, 'r')
axarr[1, 1].plot(Xnew, upper_squashed, 'r')
axarr[1, 1].plot(Xnew, lower_squashed, 'r', label="Estimated confidence intervals after squashing")
axarr[1, 1].plot(Xnew, copies[0], 'b')
axarr[1, 1].plot(Xnew, copies[1], 'b')
axarr[1, 1].plot(Xnew, copies[2], 'b', label="Squashed confidence intervals")
axarr[1, 1].set_ylim([0.0, 1.0])
axarr[1, 1].legend()


plt.show()
