from rosbags.rosbag2 import Reader, Writer
from rosbags.serde import deserialize_cdr, serialize_cdr

TOPIC_REMAP = {
    '/fisheye_front/camera_info_vf1228': '/fisheye/front/fisheye_front/camera_info',
    '/fisheye_front/image_raw': '/fisheye/front/fisheye_front/image_raw',
    '/wheel/odom': '/odom',
}

src_path = 'test2_ros2'
dst_path = 'test2_ros2_remapped'

with Reader(src_path) as reader, Writer(dst_path) as writer:
    conn_map = {}
    for conn in reader.connections:
        new_topic = TOPIC_REMAP.get(conn.topic, conn.topic)
        new_conn = writer.add_connection(new_topic, conn.msgtype)
        conn_map[conn.id] = new_conn

    for conn, timestamp, data in reader.messages():
        new_conn = conn_map[conn.id]
        writer.write(new_conn, timestamp, data)

print('Done.')