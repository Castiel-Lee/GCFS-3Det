import cv2
import json
import numpy as np
from pcdet.utils import box_utils

class Calibration(object):
    def __init__(self, calib_file, cam_name='front_center'):
        assert calib_file.split('.')[-1]=='json'
        
        with open (calib_file, 'r') as f:
            config = json.load(f)

        self.cam_name = cam_name
        self.intr_mat_undist = \
            np.asarray(config['cameras'][cam_name]['CamMatrix'])
        self.intr_mat_dist = \
            np.asarray(config['cameras'][cam_name]['CamMatrixOriginal'])
        self.dist_parms = \
            np.asarray(config['cameras'][cam_name]['Distortion'])
        self.lens = config['cameras'][cam_name]['Lens']
    
    def trans_points_from_frontcam_to_img(self, points, intrinsic_matrix):
        ''' image needs to be undistored! '''
        assert self.cam_name == 'front_center'
        trans_coor = np.array([[0, -1, 0],[0, 0, -1], [1, 0, 0]]).astype(points.dtype)
        points_camera = np.matmul(trans_coor, points.T).T
        points_trans = np.matmul(intrinsic_matrix, points_camera.T).T
        points_trans[:,0] /= points_trans[:,2]
        points_trans[:,1] /= points_trans[:,2]
        return points_trans[:,:2], points_trans[:,2]
    
    def trans_point2d_img_to_point3d_lidar(self, points_2d, depth_2d, intrinsic_matrix):
        ''' image needs to be undistored! '''
        assert self.cam_name == 'front_center'
        # trans_coor = np.array([[0, -1, 0],[0, 0, -1], [1, 0, 0]]).astype(points.dtype)
        # points_camera = np.matmul(trans_coor, points.T).T
        # points_trans = np.matmul(intrinsic_matrix, points_camera.T).T
        # points_trans[:,0] /= points_trans[:,2]
        # points_trans[:,1] /= points_trans[:,2]

        assert points_2d.shape[1] == 2 and len(depth_2d.shape) ==1
        points_2d[:,0] *= depth_2d
        points_2d[:,1] *= depth_2d
        points_2d_trans = np.concatenate([points_2d, depth_2d[..., np.newaxis]], axis=1)
        points_rect = np.matmul(np.linalg.inv(intrinsic_matrix), points_2d_trans.T).T
        trans_coor = np.array([[0, -1, 0],[0, 0, -1], [1, 0, 0]]).astype(points_rect.dtype)
        points_3d = np.matmul(np.linalg.inv(trans_coor), points_rect.T).T

        return points_3d
    
    def undistort_image_and_boxes(self, image, bboxes=None, img_normed=False):  
        # image *= 255 if img_normed else 1 
        if (self.lens == 'Fisheye'):
            undistorted_image =  cv2.fisheye.undistortImage(image, self.intr_mat_dist,\
                D=self.dist_parms, Knew=self.intr_mat_undist)
        elif (self.lens == 'Telecam'):
            undistorted_image = cv2.undistort(image, self.intr_mat_dist, \
                distCoeffs=self.dist_parms, newCameraMatrix=self.intr_mat_undist)
        else:
            NotImplementedError('Unrecognized Type of Len: %s' % self.lens)
        
        # undistorted_image /= 255 if img_normed else 1

        if bboxes is None:
            undistorted_bboxes = bboxes
        else:
            undistorted_bboxes = []
            for bbox in bboxes:
                # Define the corner points of the bounding box
                corners = np.array([
                    [bbox[0], bbox[1]],  # Top-left
                    [bbox[2], bbox[1]],  # Top-right
                    [bbox[2], bbox[3]],  # Bottom-right
                    [bbox[0], bbox[3]]   # Bottom-left
                ], dtype=np.float32).reshape(-1, 1, 2)

                # Undistort the points using both the original and new camera matrices
                undistorted_corners = cv2.undistortPoints(corners, self.intr_mat_dist, self.dist_parms, P=self.intr_mat_undist)
                undistorted_corners = undistorted_corners.reshape(-1, 2)

                # Calculate the new bounding box in the undistorted image
                x_min, y_min = undistorted_corners.min(axis=0)
                x_max, y_max = undistorted_corners.max(axis=0)
                undistorted_bboxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
            undistorted_bboxes = np.array(undistorted_bboxes)
        
        return undistorted_image, undistorted_bboxes

    def undistort_boxes(self, bboxes):  

        undistorted_bboxes = []
        for bbox in bboxes:
            # Define the corner points of the bounding box
            corners = np.array([
                [bbox[0], bbox[1]],  # Top-left
                [bbox[2], bbox[1]],  # Top-right
                [bbox[2], bbox[3]],  # Bottom-right
                [bbox[0], bbox[3]]   # Bottom-left
            ], dtype=np.float32).reshape(-1, 1, 2)

            # Undistort the points using both the original and new camera matrices
            undistorted_corners = cv2.undistortPoints(corners, self.intr_mat_dist, self.dist_parms, P=self.intr_mat_undist)
            undistorted_corners = undistorted_corners.reshape(-1, 2)

            # Calculate the new bounding box in the undistorted image
            x_min, y_min = undistorted_corners.min(axis=0)
            x_max, y_max = undistorted_corners.max(axis=0)
            undistorted_bboxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
        undistorted_bboxes = np.array(undistorted_bboxes)
        
        return undistorted_bboxes

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_img, pts_depth = self.trans_points_from_frontcam_to_img(pts_lidar, self.intr_mat_undist)
        return pts_img, pts_depth
    
    def img_to_lidar(self, points_2d, depth_2d):
        return self.trans_point2d_img_to_point3d_lidar(points_2d, depth_2d, self.intr_mat_undist)

    def boxes3d_lidar_to_boxes2d_img(self, boxes_3d, image_shape):

        corners3d = box_utils.boxes_to_corners_3d(boxes_3d)
        pts_img, _ = self.lidar_to_img(corners3d.reshape(-1, 3))
        corners_in_image = pts_img.reshape(-1, 8, 2)

        min_uv = np.min(corners_in_image, axis=1)  # (N, 2)
        max_uv = np.max(corners_in_image, axis=1)  # (N, 2)
        boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
        if image_shape is not None:
            boxes2d_image[:, 0] = np.clip(boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_image[:, 1] = np.clip(boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
            boxes2d_image[:, 2] = np.clip(boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_image[:, 3] = np.clip(boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

        return boxes2d_image
    
    def img_boxes_constraint(self, boxes2d_image, image_shape):

        boxes2d_image[:, 0] = np.clip(boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 1] = np.clip(boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
        boxes2d_image[:, 2] = np.clip(boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 3] = np.clip(boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

        return boxes2d_image

