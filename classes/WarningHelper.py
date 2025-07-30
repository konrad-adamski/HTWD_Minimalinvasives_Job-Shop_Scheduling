from typing import Dict, List

def warn_missing_routing_operations(missing: Dict[str, List[int]]):
    print("[Warning] Some routing operations are missing:")
    for routing_id, seqs in missing.items():
        seqs_str = ", ".join(str(s) for s in sorted(seqs))
        print(f"  - Routing '{routing_id}' is missing these operations: {seqs_str}")
