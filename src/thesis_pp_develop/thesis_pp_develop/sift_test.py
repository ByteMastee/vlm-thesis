import cv2
import os

# Point this to one of your saved frame images
frames_dir = '/root/UVC_ws/vf_robot_model_ros2/yolo_frames3'
frame_file = os.path.join(frames_dir, 'frame_00420.png')

img = cv2.imread(frame_file)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kps, descs = sift.detectAndCompute(gray, None)

print(f'SIFT keypoints on full image: {len(kps)}')

# Draw keypoints
out = cv2.drawKeypoints(img, kps, None,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('/root/UVC_ws/vf_robot_model_ros2/pp_tunning/sift_test.png', out)
print('Saved to sift_test.png')