import os
import json


def node_to_dict(node):
    """Recursively convert a node and its children into a dictionary for JSON serialization."""
    return {
        "player": node.state.current_player,
        "board": ''.join(node.state.board),
        "visits": node.visits,
        "wins": node.wins,
        "children": [node_to_dict(child) for child in node.children]
    }


def save_tree_state(root, iteration: int, out_dir: str = "mcts_tictactoe/tree_logs"):
    """
    Save the tree structure starting from root to a JSON file for visualization.

    Args:
        root: the root node of the current MCTS state.
        iteration: which iteration this snapshot corresponds to.
        out_dir: directory where JSON files are saved.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    snapshot = node_to_dict(root)

    file_path = os.path.join(out_dir, f"tree_iter_FINAL.json")
    with open(file_path, 'w') as f:
        json.dump(snapshot, f, indent=2)

    print(f"[INFO] Tree snapshot saved to {file_path}")