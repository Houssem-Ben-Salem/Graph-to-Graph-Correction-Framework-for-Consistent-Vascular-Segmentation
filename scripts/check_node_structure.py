#!/usr/bin/env python
"""Check how spatial data is stored in nodes"""

import sys
import pickle
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("🔍 Checking Node Structure\n")
    
    # Load one example
    patient_id = "PA000005"
    gt_file = Path(f"extracted_graphs/{patient_id}/{patient_id}_GT.pkl")
    
    with open(gt_file, 'rb') as f:
        gt_graph = pickle.load(f)
    
    print(f"Total nodes: {len(gt_graph.nodes)}")
    
    # Check first few nodes
    for i in range(min(3, len(gt_graph.nodes))):
        node = gt_graph.nodes[i]
        print(f"\nNode {i}:")
        print(f"  Type: {type(node)}")
        print(f"  Content: {node}")
        
        # If it's an object, check its attributes
        if hasattr(node, '__dict__'):
            print(f"  Attributes: {node.__dict__}")
        elif hasattr(node, '_fields'):  # namedtuple
            print(f"  Fields: {node._fields}")
            for field in node._fields:
                print(f"    {field}: {getattr(node, field)}")

if __name__ == "__main__":
    main()