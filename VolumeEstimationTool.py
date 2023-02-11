import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

def fit_ellipse(pointcloud, x_axis_length, isCentered=True, min_aspect_ratio=0.5, max_aspect_ratio=1):
    # Define the cost function to minimize
    def cost_function(params):
        x0, y0, a, b = params
        ellipse = ((pointcloud[:, 0] - x0) / a)**2 + ((pointcloud[:, 1] - y0) / b)**2
        return np.sum((ellipse - 1)**2)

    # Define the initial guess
    x0 = np.mean(pointcloud[:, 0])
    y0 = np.mean(pointcloud[:, 1])
    a = x_axis_length / 2
    b = np.sqrt(np.sum((pointcloud[:, 1] - y0)**2) / len(pointcloud))

    # Define the bounds for the optimization
    if (isCentered):
        bounds = [(-a, a), (-2*a, 2*a), (a-5, a+5), (min_aspect_ratio*a, max_aspect_ratio*a)]
    else:
        bounds = [(None, None), (None, None), (a-5, a+5), (min_aspect_ratio*a, max_aspect_ratio*a)]

    # Minimize the cost function
    result = minimize(cost_function, [x0, y0, a, b], bounds=bounds)

    # Return the parameters that define the best fitting ellipse
    x0, y0, a, b = result.x
    theta = 0.0
    return x0, y0, a, b, theta


def split_pointcloud(pointcloud, tol=2):
    # Sort the point cloud by x-coordinate
    pointcloud = pointcloud[pointcloud[:, 0].argsort()]

    # Split the point cloud into 2D profiles
    profiles = []
    current_profile = []
    prev_x = None
    for i in range(len(pointcloud)):
        x, y, z = pointcloud[i]
        if prev_x is None or abs(x - prev_x) < tol:
            # Add point to current profile
            current_profile.append([y, z])
        else:
            # Start new profile
            profiles.append(np.array(current_profile))
            current_profile = [[y, z]]
        prev_x = x

    # Add last profile to list
    if current_profile:
        profiles.append(np.array(current_profile))

    return profiles


#Vars
DRAW_PROFILES = True
PERFORM_PCA = True
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1

# Load point cloud from file and truncate decimals
# pointcloud = np.loadtxt('pts_1perfil.xyz', delimiter=' ', usecols=(1,2), dtype=int)
pointcloud = np.loadtxt('pts.xyz', delimiter=' ', usecols=(0,1,2), dtype=int)

# use the function split_pointcloud to split the pointcloud in profiles
profiles = split_pointcloud(pointcloud)

# plotting
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
ax1.set_aspect('equal')
ax2.set_aspect('equal')
fig2, axs = plt.subplots(nrows=len(profiles), ncols=2, figsize=(10,10))

#do a for loop to do the code bellow for each profile
for i, profile in enumerate(profiles):

    # Perform PCA to align all profiles
    if(PERFORM_PCA):

        # Perform PCA   
        pca = PCA(n_components=2)
        pca.fit(profile)

        # Get eigenvalues and eigenvectors
        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_

        # Project point cloud onto principal components
        # projected = pca.transform(profile)

        # move profile to the origin    
        projected = profile - np.mean(profile, axis=0)
        # projected = projected - np.mean(projected, axis=1)


    else:

        projected = profile

    # calculate the range of the data in the proyection plane
    x_min = projected[:, 0].min()
    x_max = projected[:, 0].max()
    y_min = projected[:, 1].min()
    y_max = projected[:, 1].max()

    # get 4 points of the roi
    roi = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])

    # define a variable as the horizontal_spread of the roi
    horizontal_spread = x_max - x_min

    # define a variable as the vertical_spread of the roi
    vertical_spread = y_max - y_min

    # Fit ellipse to projected point cloud
    ellipse = fit_ellipse(projected, horizontal_spread, isCentered=PERFORM_PCA, min_aspect_ratio=MIN_ASPECT_RATIO, max_aspect_ratio=MAX_ASPECT_RATIO)

    # Plot original point cloud
    ax1.scatter(profile[:, 0], profile[:, 1] + i*50)
    ax1.set_title('Original Point Cloud')

    # Plot projected point cloud
    ax2.scatter(projected[:, 0], projected[:, 1] + i*50)
    ax2.set_title('Projected Point Cloud')

    #add ellipse to the plot
    # Generate x and y values for the ellipse
    center = (ellipse[0], ellipse[1])
    a = ellipse[2]
    b = ellipse[3]
    angle = 0
    t = np.linspace(0, 2*np.pi, 100)
    x = center[0] + a*np.cos(t)*np.cos(angle) - b*np.sin(t)*np.sin(angle)
    y = center[1] + a*np.cos(t)*np.sin(angle) + b*np.sin(t)*np.cos(angle)

    # Plot the ellipse
    ax2.plot(x, y + i*50, color='r', linestyle='--', linewidth=1)

    if(DRAW_PROFILES):
        
        # Plot original point cloud
        axs[i, 0].scatter(profile[:, 0], profile[:, 1])
        axs[i, 0].set_title('Profile {}'.format(i+1))

        # Plot projected point cloud
        axs[i, 1].scatter(projected[:, 0], projected[:, 1])
        axs[i, 1].set_title('Projected Profile {}'.format(i+1))

        # Plot the ellipse
        axs[i, 1].plot(x, y, color='r', linestyle='--', linewidth=1)

        # Plot the x-axis of the ellipse
        axs[i, 1].plot([center[0]-a, center[0]+a], [center[1], center[1]], color='b', linestyle='-', linewidth=1)

        axs[i, 0].set_aspect('equal')
        axs[i, 1].set_aspect('equal')

# Show plot
plt.show()



