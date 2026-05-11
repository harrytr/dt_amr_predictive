#!/usr/bin/env python3
import json
from tasks import TASK_REGISTRY

tasks = []
for task_id, task in TASK_REGISTRY.items():
    tasks.append({
        "id": task_id,
        "name": getattr(task, "name", task_id),
        "is_classification": getattr(task, "is_classification", False),
    })

print(json.dumps(tasks))
