#!/usr/bin/env python
"""Debug real GT/PRED graph pairs"""

import sys
import pickle
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("🔍 Debugging Real GT/PRED Graph Pairs\n")
    
    # Check first few patients
    extracted_dir = Path("extracted_graphs")
    patients = ["PA000005", "PA000016", "PA000024"]
    
    for patient_id in patients:
        print(f"=== {patient_id} ===")
        
        patient_dir = extracted_dir / patient_id
        gt_file = patient_dir / f"{patient_id}_GT.pkl"
        pred_file = patient_dir / f"{patient_id}_PRED.pkl"
        
        if not gt_file.exists():
            print(f"❌ GT file missing: {gt_file}")
            continue
            
        if not pred_file.exists():
            print(f"❌ PRED file missing: {pred_file}")
            continue
        
        try:
            # Load GT graph
            with open(gt_file, 'rb') as f:
                gt_graph = pickle.load(f)
            
            # Load PRED graph  
            with open(pred_file, 'rb') as f:
                pred_graph = pickle.load(f)
            
            print(f"GT Graph:")
            print(f"  Nodes: {len(gt_graph.nodes)}")
            print(f"  Edges: {len(gt_graph.edges)}")
            if hasattr(gt_graph, 'node_positions') and gt_graph.node_positions:
                positions = list(gt_graph.node_positions.values())[:3]
                print(f"  Sample positions: {positions}")
            
            print(f"PRED Graph:")
            print(f"  Nodes: {len(pred_graph.nodes)}")
            print(f"  Edges: {len(pred_graph.edges)}")
            if hasattr(pred_graph, 'node_positions') and pred_graph.node_positions:
                positions = list(pred_graph.node_positions.values())[:3]
                print(f"  Sample positions: {positions}")
            
            # Check if graphs have spatial information
            gt_has_positions = hasattr(gt_graph, 'node_positions') and gt_graph.node_positions
            pred_has_positions = hasattr(pred_graph, 'node_positions') and pred_graph.node_positions
            
            print(f"Spatial Info:")
            print(f"  GT has positions: {gt_has_positions}")
            print(f"  PRED has positions: {pred_has_positions}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error loading {patient_id}: {e}")
            print()

if __name__ == "__main__":
    main()