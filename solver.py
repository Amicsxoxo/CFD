import numpy as np
from matplotlib import pyplot

#The number of iterations we are going to plot, use lower values (<5) for a smoother simulation and higher (>50) values for a faster simulation
plot_every = 100

def distance(x1, y1, x2, y2):
  return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 )

def main():
  #Number of pixels in the x and y planes
  nX = 400
  nY = 100
  tau = .53
  #Number of iteration
  nT = 6000

  #Lattice speeds and weights
  nL = 9
  cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
  cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])

  #The weights that relate the lattice boltzman method to the navier stokes equation its the D2Q9
  weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

  #Intial conditions
  F = np.ones((nY, nX, nL)) + .01*np.random.randn(nY, nX, nL)
  #Giving the right node a non zero value so that the flow moves to the right, it denotes at every single lattice, at every single cell give the third node a value of 2.3
  F[:, :, 3] = 2.3

  #False defines it as free space and not a boundary
  cylinder = np.full((nY, nX), False)

  for y in range(0, nY):
    for x in range(0, nX):
      #This is the boundary conditions can be changed
      if (distance(nX//4, nY//2, x, y) < 13):
        #Y comes first so it's the 0 axis, then X is the 1 axis
        cylinder[y][x] = True

  #Main loop
  for it in range(nT):
    print(it)

    #Zu hae boundary it makes the velocity around the boundary to be absorbed and not be bounced back
    F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]
    F[:, 0, [2, 3, 4]] =  F[:, 1, [2, 3, 4]]


    for i, cx, cy in zip(range(nL), cxs, cys):
      #X is the 1 axis and Y is the 0 axis
      F[:, :, i] = np.roll(F[:, :, i], cx, axis = 1)
      F[:, :, i] = np.roll(F[:, :, i], cy, axis = 0)

    #We set all the lattice in our boundary to their opposite velocities respectively
    bndryF = F[cylinder, :]
    bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

    #Fluid variables, desity and velocity in the x and y directions
    rho = np.sum(F, 2)
    ux = np.sum(F* cxs, 2) / rho
    uy = np.sum(F* cys, 2) / rho

    #Made sure there is no fluid movement in the solid
    F[cylinder, :] = bndryF
    ux[cylinder] = 0
    uy[cylinder] = 0

    #Collision
    #Made Feq an array of the same size as F
    Feq = np.zeros(F.shape)
    for i, cx, cy, w in zip(range(nL), cxs, cys, weights):
      Feq[:, :, i] = rho * w * (
        1 + 3 * (cx * ux + cy * uy) + 9 * (cx * ux + cy * uy) ** 2 / 2 - 3 * ( ux** 2 + uy **2)/2
      ) 
    F = F + -(1/tau) * (F-Feq) 

    #Plotting the functions
    if (it%plot_every == 0 ):
      #Calculating the curl
      dfydx = ux[2:, 1: -1] - ux[0: -2, 1: -1]
      dfxdy = uy[1: -1, 2:] - uy[1: -1, 0: -2]
      curl = dfydx - dfxdy
      pyplot.title("LBM Cfd Solver")
      #Simulating the curl
      #Used the colour map of bwr which gives blue to the negative values and red to positve values
      pyplot.imshow(curl, cmap = "bwr")
      # #Simulating the velocity 
      # pyplot.imshow(np.sqrt(ux**2 + uy**2))
      pyplot.pause(.01)
      pyplot.cla()


if __name__ == "__main__":
  main()