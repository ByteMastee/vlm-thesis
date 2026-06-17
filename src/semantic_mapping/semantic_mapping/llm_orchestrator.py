import json
import threading
import requests

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

# ─────────────────────────────────────────────
# USER INPUTS
# ─────────────────────────────────────────────
OLLAMA_URL  = "http://172.17.0.1:11434/api/chat"
MODEL_NAME  = "llama3.2:3b"
OFFSET_M    = 0.25
GOAL_TOPIC  = '/goal_pose'
GOAL_FRAME  = 'map'
# ─────────────────────────────────────────────


class LLMOrchestratorNode(Node):
    def __init__(self):
        super().__init__('llm_orchestrator_node')

        self.semantic_map  = None
        self.pipeline_name = None
        self.map_received  = threading.Event()

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.create_subscription(
            String,
            '/semantic_map_json',
            lambda msg: self._map_cb(msg, 'YOLO+VLM'),
            latched_qos
        )

        self.create_subscription(
            String,
            '/vit_semantic_map_json',
            lambda msg: self._map_cb(msg, 'ViT+VLM'),
            latched_qos
        )

        self.goal_pub = self.create_publisher(PoseStamped, GOAL_TOPIC, 10)

        self.get_logger().info('LLM Orchestrator Node started.')
        self.get_logger().info('Waiting for semantic map on /semantic_map_json or /vit_semantic_map_json ...')

        thread = threading.Thread(target=self._input_loop, daemon=True)
        thread.start()

    def _map_cb(self, msg, pipeline_name):
        try:
            self.semantic_map  = json.loads(msg.data)
            self.pipeline_name = pipeline_name
            self.get_logger().info(
                f'Semantic map received from [{pipeline_name}] — '
                f'{len(self.semantic_map)} objects: {list(self.semantic_map.keys())}'
            )
            self.map_received.set()
        except Exception as e:
            self.get_logger().error(f'Failed to parse semantic map JSON: {e}')

    def _input_loop(self):
        self.get_logger().info('Waiting for semantic map before accepting instructions...')
        self.map_received.wait()
        self.get_logger().info(f'Map ready [{self.pipeline_name}]. You can now enter navigation instructions.')
        print()

        while rclpy.ok():
            try:
                instruction = input("Enter navigation instruction (or 'quit' to exit): ").strip()
            except EOFError:
                break

            if instruction.lower() == 'quit':
                break
            if not instruction:
                continue

            print("Querying LLM...")
            prompt       = self._build_prompt(self.semantic_map, instruction)
            raw_response = self._query_llm(prompt)

            print(f"\nLLM Response:\n{raw_response}\n")

            object_name, direction, goal_x, goal_y = self._parse_response(raw_response)

            if object_name is None and goal_x is None:
                print("No matching object found in the semantic map for this instruction.")
            elif goal_x is not None and goal_y is not None:
                if direction:
                    goal_x, goal_y = self._apply_offset(goal_x, goal_y, direction, OFFSET_M)
                    print(f"Matched Object : {object_name}")
                    print(f"Direction      : {direction}")
                    print(f"Goal Coordinate: x={round(goal_x, 4)}, y={round(goal_y, 4)}")
                else:
                    print(f"Matched Object : {object_name}")
                    print(f"Goal Coordinate: x={goal_x}, y={goal_y}")

                self._publish_goal(goal_x, goal_y)
            else:
                print("Could not parse goal coordinate from LLM response.")
            print()

    def _publish_goal(self, goal_x, goal_y):
        msg                        = PoseStamped()
        msg.header.stamp           = self.get_clock().now().to_msg()
        msg.header.frame_id        = GOAL_FRAME
        msg.pose.position.x        = float(goal_x)
        msg.pose.position.y        = float(goal_y)
        msg.pose.position.z        = 0.0
        msg.pose.orientation.w     = 1.0
        self.goal_pub.publish(msg)
        self.get_logger().info(
            f'Goal published to {GOAL_TOPIC} — x={goal_x}, y={goal_y}'
        )

    def _build_prompt(self, semantic_map, instruction):
        map_lines = []
        for label, data in semantic_map.items():
            map_lines.append(f"  - {label}: x={data['x']}, y={data['y']}")
        map_str = "\n".join(map_lines)

        prompt = f"""You are a robot navigation assistant.
You have a semantic map of the environment with the following objects and their 2D positions (in metres):

{map_str}

The robot has received this instruction: "{instruction}"

Your task:
1. Identify which object in the map best matches the instruction.
2. Only use objects that exist in the map above. Do not invent or assume any object not listed.
3. If no object in the map matches the instruction, respond exactly with: NOT_FOUND
4. If the instruction mentions a relative direction (in front of, behind, left of, right of), extract the direction and the object separately.
5. Otherwise, return the object name and its coordinates directly.

Respond in one of these exact formats and nothing else:

If object is found with no relative direction:
OBJECT: <matched object name>
GOAL: x=<value>, y=<value>

If object is found with a relative direction:
OBJECT: <matched object name>
DIRECTION: <front|behind|left|right>
GOAL: x=<value>, y=<value>

If not found:
NOT_FOUND
"""
        return prompt

    def _query_llm(self, prompt):
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["message"]["content"]

    def _apply_offset(self, x, y, direction, offset):
        if direction == "front":
            return x + offset, y
        elif direction == "behind":
            return x - offset, y
        elif direction == "left":
            return x, y + offset
        elif direction == "right":
            return x, y - offset
        return x, y

    def _parse_response(self, response_text):
        response_text = response_text.strip()

        if "NOT_FOUND" in response_text.replace("_", "").replace(" ", ""):
            return None, None, None, None

        object_name = None
        direction   = None
        goal_x      = None
        goal_y      = None

        for line in response_text.splitlines():
            line = line.strip()
            if line.startswith("OBJECT:"):
                object_name = line.split("OBJECT:")[-1].strip()
            elif line.startswith("DIRECTION:"):
                direction = line.split("DIRECTION:")[-1].strip().lower()
            elif line.startswith("GOAL:"):
                goal_part = line.split("GOAL:")[-1].strip()
                for part in goal_part.split(","):
                    part = part.strip()
                    if part.startswith("x="):
                        goal_x = float(part.split("=")[-1].strip())
                    elif part.startswith("y="):
                        goal_y = float(part.split("=")[-1].strip())

        return object_name, direction, goal_x, goal_y


def main(args=None):
    rclpy.init(args=args)
    node = LLMOrchestratorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()




# import json
# import requests

# # ─────────────────────────────────────────────
# # USER INPUTS
# # ─────────────────────────────────────────────
# MAP_JSON_PATH = "/root/UVC_ws/vf_robot_model_ros2/Final_Output/Thesis_RWvit/VITp1_1/VITp1_1_vit_vlm_object_stack.json"
# OLLAMA_URL    = "http://172.17.0.1:11434/api/chat"
# MODEL_NAME    = "llama3.2:3b"
# OFFSET_M      = 0.25
# # ─────────────────────────────────────────────


# def load_map(path):
#     with open(path, "r") as f:
#         return json.load(f)


# def build_prompt(semantic_map, instruction):
#     map_lines = []
#     for label, data in semantic_map.items():
#         map_lines.append(f"  - {label}: x={data['x']}, y={data['y']}")
#     map_str = "\n".join(map_lines)

#     prompt = f"""You are a robot navigation assistant.
# You have a semantic map of the environment with the following objects and their 2D positions (in metres):

# {map_str}

# The robot has received this instruction: "{instruction}"

# Your task:
# 1. Identify which object in the map best matches the instruction.
# 2. Only use objects that exist in the map above. Do not invent or assume any object not listed.
# 3. If no object in the map matches the instruction, respond exactly with: NOT_FOUND
# 4. If the instruction mentions a relative direction (in front of, behind, left of, right of), extract the direction and the object separately.
# 5. Otherwise, return the object name and its coordinates directly.

# Respond in one of these exact formats and nothing else:

# If object is found with no relative direction:
# OBJECT: <matched object name>
# GOAL: x=<value>, y=<value>

# If object is found with a relative direction:
# OBJECT: <matched object name>
# DIRECTION: <front|behind|left|right>
# GOAL: x=<value>, y=<value>

# If not found:
# NOT_FOUND
# """
#     return prompt


# def query_llm(prompt):
#     payload = {
#         "model": MODEL_NAME,
#         "messages": [{"role": "user", "content": prompt}],
#         "stream": False
#     }
#     response = requests.post(OLLAMA_URL, json=payload, timeout=60)
#     response.raise_for_status()
#     return response.json()["message"]["content"]


# def apply_offset(x, y, direction, offset):
#     if direction == "front":
#         return x + offset, y
#     elif direction == "behind":
#         return x - offset, y
#     elif direction == "left":
#         return x, y + offset
#     elif direction == "right":
#         return x, y - offset
#     return x, y


# def parse_response(response_text, semantic_map):
#     response_text = response_text.strip()

#     if "NOT_FOUND" in response_text.replace("_", "").replace(" ", ""):
#         return None, None, None, None

#     object_name = None
#     direction   = None
#     goal_x      = None
#     goal_y      = None

#     for line in response_text.splitlines():
#         line = line.strip()
#         if line.startswith("OBJECT:"):
#             object_name = line.split("OBJECT:")[-1].strip()
#         elif line.startswith("DIRECTION:"):
#             direction = line.split("DIRECTION:")[-1].strip().lower()
#         elif line.startswith("GOAL:"):
#             goal_part = line.split("GOAL:")[-1].strip()
#             for part in goal_part.split(","):
#                 part = part.strip()
#                 if part.startswith("x="):
#                     goal_x = float(part.split("=")[-1].strip())
#                 elif part.startswith("y="):
#                     goal_y = float(part.split("=")[-1].strip())

#     return object_name, direction, goal_x, goal_y


# def main():
#     print("Loading semantic map...")
#     semantic_map = load_map(MAP_JSON_PATH)
#     print(f"Loaded {len(semantic_map)} objects: {list(semantic_map.keys())}")
#     print()

#     while True:
#         instruction = input("Enter navigation instruction (or 'quit' to exit): ").strip()
#         if instruction.lower() == "quit":
#             break
#         if not instruction:
#             continue

#         print("Querying LLM...")
#         prompt = build_prompt(semantic_map, instruction)
#         raw_response = query_llm(prompt)

#         print(f"\nLLM Response:\n{raw_response}\n")

#         object_name, direction, goal_x, goal_y = parse_response(raw_response, semantic_map)

#         if object_name is None and goal_x is None:
#             print("No matching object found in the semantic map for this instruction.")
#         elif goal_x is not None and goal_y is not None:
#             if direction:
#                 goal_x, goal_y = apply_offset(goal_x, goal_y, direction, OFFSET_M)
#                 print(f"Matched Object : {object_name}")
#                 print(f"Direction      : {direction}")
#                 print(f"Goal Coordinate: x={round(goal_x, 4)}, y={round(goal_y, 4)}")
#             else:
#                 print(f"Matched Object : {object_name}")
#                 print(f"Goal Coordinate: x={goal_x}, y={goal_y}")
#         else:
#             print("Could not parse goal coordinate from LLM response.")
#         print()


# if __name__ == "__main__":
#     main()


