import os
import json

lit_dir = "C:/d_drive/projects/Project1/research/literature"
files = [f for f in os.listdir(lit_dir) if f.endswith('.json') and not f.startswith('_') and not f.startswith('__')]

count = 0
output = []
for f in files:
    try:
        with open(os.path.join(lit_dir, f), 'r', encoding='utf-8') as file:
            data = json.load(file)
            title = data.get("title", "")
            abstract = data.get("abstract", "")
            output.append(f"**{count+1}. {title}**\n**Important Findings:** {abstract}\n")
            count += 1
            if count == 15:
                break
    except:
        pass

with open("c:/d_drive/projects/Project1/full_findings.txt", "w", encoding='utf-8') as f:
    f.write("\n".join(output))
