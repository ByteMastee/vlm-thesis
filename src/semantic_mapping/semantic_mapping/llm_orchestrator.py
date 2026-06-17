import json
import requests

# ─────────────────────────────────────────────
# USER INPUTS
# ─────────────────────────────────────────────
MAP_JSON_PATH = "/root/UVC_ws/vf_robot_model_ros2/Final_Output/Thesis_RWvit/VITp1_1/VITp1_1_vit_vlm_object_stack.json"
OLLAMA_URL    = "http://172.17.0.1:11434/api/chat"
MODEL_NAME    = "llama3.2:3b"
OFFSET_M      = 0.25
# ─────────────────────────────────────────────


def load_map(path):
    with open(path, "r") as f:
        return json.load(f)


def build_prompt(semantic_map, instruction):
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


def query_llm(prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["message"]["content"]


def apply_offset(x, y, direction, offset):
    if direction == "front":
        return x + offset, y
    elif direction == "behind":
        return x - offset, y
    elif direction == "left":
        return x, y + offset
    elif direction == "right":
        return x, y - offset
    return x, y


def parse_response(response_text, semantic_map):
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


def main():
    print("Loading semantic map...")
    semantic_map = load_map(MAP_JSON_PATH)
    print(f"Loaded {len(semantic_map)} objects: {list(semantic_map.keys())}")
    print()

    while True:
        instruction = input("Enter navigation instruction (or 'quit' to exit): ").strip()
        if instruction.lower() == "quit":
            break
        if not instruction:
            continue

        print("Querying LLM...")
        prompt = build_prompt(semantic_map, instruction)
        raw_response = query_llm(prompt)

        print(f"\nLLM Response:\n{raw_response}\n")

        object_name, direction, goal_x, goal_y = parse_response(raw_response, semantic_map)

        if object_name is None and goal_x is None:
            print("No matching object found in the semantic map for this instruction.")
        elif goal_x is not None and goal_y is not None:
            if direction:
                goal_x, goal_y = apply_offset(goal_x, goal_y, direction, OFFSET_M)
                print(f"Matched Object : {object_name}")
                print(f"Direction      : {direction}")
                print(f"Goal Coordinate: x={round(goal_x, 4)}, y={round(goal_y, 4)}")
            else:
                print(f"Matched Object : {object_name}")
                print(f"Goal Coordinate: x={goal_x}, y={goal_y}")
        else:
            print("Could not parse goal coordinate from LLM response.")
        print()


if __name__ == "__main__":
    main()


