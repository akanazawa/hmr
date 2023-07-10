"""
Module with logging utilities
"""

import logging

import matplotlib.pyplot as plt
import vlogging

import src.util.renderer

def log_mesh_prediction(
        logger: logging.getLogger, image_path, image, preprocessing_parameters: dict,
        joints, vertices, camera, smpl_face_path):

    cam_for_render, vert_shifted, joints_orig = src.util.renderer.get_original(
        proc_param=preprocessing_parameters,
        verts=vertices,
        cam=camera,
        joints=joints,
        img_size=image.shape[:2]
    )

    # Render results
    skel_img = src.util.renderer.draw_skeleton(image, joints_orig)

    renderer = src.util.renderer.SMPLRenderer(face_path=smpl_face_path)

    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=image, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=image.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=image.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=image.shape[:2])

    figure = plt.figure()

    plt.clf()
    plt.subplot(231)
    plt.imshow(image)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()

    logger.info(
        vlogging.VisualRecord(image_path, figure)
    )
