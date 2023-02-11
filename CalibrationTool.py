import open3d as o3d
from scipy.spatial.transform import Rotation as R
import numpy as np
import json

# params
export_matrix = False

# origin 
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=80, origin=[0,0,0])

# Load two mesh from ply file
pcd = o3d.io.read_point_cloud(r'.\PointClouds\pc0.ply')
pcd2 = o3d.io.read_point_cloud(r'.\PointClouds\pc1.ply')

pcd.paint_uniform_color([1, 0.706, 0])
pcd2.paint_uniform_color([1, 0.0, 0.0])

# # draw
# o3d.visualization.draw_geometries([pcd, pcd2,  origin])

# create matrix

matrix = np.eye(4)

# translation
matrix[0,3] = 0.0
matrix[1,3] = 0.0
matrix[2,3] = 0.0

# rotation
r = R.from_euler('xyz', [0.3, 33, 0], degrees=True)
matrix[:3,:3] = r.as_matrix()

#final matrix
matrix_pcd = np.eye(4)
matrix_pcd = matrix

# prealign
pcd.transform(matrix_pcd)

# get points
points = np.asarray(pcd.points)

# get values upper than z=500
# points = points[points[:,2] < 500]

pcd.points = o3d.utility.Vector3dVector(points)

pcd.paint_uniform_color([1, 0.706, 0])

print(matrix_pcd)

# create matrix
matrix = np.eye(4)

# translation
matrix[0,3] = 712.0
matrix[1,3] = 0.0
matrix[2,3] = 31.0

# rotation
r = R.from_euler('xyz', [-1.05, -39.15, 0], degrees=True)
matrix[:3,:3] = r.as_matrix()

# get points
points = np.asarray(pcd2.points)

# get values upper than z=500
# points = points[points[:,2] < 500]

pcd2.points = o3d.utility.Vector3dVector(points)

#final matrix
matrix_pcd2 = np.eye(4)
matrix_pcd2 = matrix

# prealign
pcd2.transform(matrix_pcd2)

print(matrix_pcd2)

o3d.visualization.draw_geometries([pcd, pcd2, origin])


# icp registration
# reg_p2p = o3d.pipelines.registration.registration_icp(pcd, pcd2, 100, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint(False))

# Apply transformation
# pcd2.transform(reg_p2p.transformation)


# show the result
# o3d.visualization.draw_geometries([pcd, pcd2, origin])


# exporting final matrix

if(export_matrix):
    # Convert numpy matrix to list
    matrix_list = matrix_pcd.tolist()

    # Write list to json file
    with open('.\Transformations\matrix_VisionSystem1.json', 'w') as f:
        json.dump(matrix_list, f)

        
    # Convert numpy matrix to list
    matrix_list = matrix_pcd2.tolist()

    # Write list to json file
    with open('.\Transformations\matrix_VisionSystem2.json', 'w') as f:
        json.dump(matrix_list, f)