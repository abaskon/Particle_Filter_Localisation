import numpy as np
import plotly.express as px

#Probablity of finding robot follows a gaussian distribution with mean as 0 and standard deviation a high value
#uniformly randomise x,y values over the coordinate frame....take theta values lite for now
#the mean value gives the position of the robot
N=300
mean = np.array([0,0])
covariance = np.diag([15,15])
samples = np.random.multivariate_normal(mean, covariance, size=N)

#NOTE : samples store the x and y co ordinate of the points


#Plot the particles throughout the world following this distribution
fig = px.scatter(x=samples[:,0], y=samples[:,1])
fig.update_layout(width = 500, height = 500, title = " Points ")
fig.update_yaxes(range=[-10,10], scaleanchor = "x",scaleratio = 1) # axis equal
fig.show()

#Update with information from a measurement
#sensor that only give us information about or x position with Gaussian noise with standard deviation sigma
#m is the sensor measurement
#implement this with  m=3  and  σ=1 :
#weight of each point is the likelihood function
m=3 # measurement
sigma=1 # standard deviation
variance = sigma**2
def likelihood(location):
  print(location[0])
  return np.exp(-0.5*(location[0]-m)**2/variance)

weights =  likelihood(samples[0])


weights = np.apply_along_axis(likelihood, 1, samples)

fig = px.scatter(x=samples[:,0], y=samples[:,1], size=weights)
fig.update_layout(width = 500, height = 500, title = "Weighted samples from posterior")
fig.update_yaxes(range=[-10,10], scaleanchor = "x",scaleratio = 1) # axis equal
fig.show()

# Simulate all samples forward for one second, using 10 Euler steps:
V=10
predictions = np.copy(samples)
for i in range(10):
  x = predictions[:,0]
  y = predictions[:,1]
  norm = np.sqrt(x**2 + y**2)
  predictions[:,0] -= 0.1*y*V/norm
  predictions[:,1] += 0.1*x*V/norm



fig = px.scatter(x=predictions[:,0], y=predictions[:,1], size=weights)
fig.update_layout(width = 500, height = 500, title = "Weighted samples from posterior")
fig.update_yaxes(range=[-10,10], scaleanchor = "x",scaleratio = 1) # axis equal
fig.show()

sample_indices = np.random.choice(len(samples),p=weights/np.sum(weights),size=N)
samples = predictions[sample_indices]

fig = px.scatter(x=samples[:,0], y=samples[:,1])
fig.update_layout(width = 500, height = 500, title = "Reweighted samples")
fig.update_yaxes(range=[-10,10], scaleanchor = "x",scaleratio = 1) # axis equal
fig.show()

m=5 # measurement
def likelihood(location): return np.exp(-0.5*(location[0]-m)**2/variance)

weights = np.apply_along_axis(likelihood, 1, samples)

fig = px.scatter(x=samples[:,0], y=samples[:,1], size=weights)
fig.update_layout(width = 500, height = 500, title = "Weighted samples from posterior")
fig.update_yaxes(range=[-10,10], scaleanchor = "x",scaleratio = 1) # axis equal
fig.show()
