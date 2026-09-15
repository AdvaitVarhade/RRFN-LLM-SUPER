import os
import json

lit_dir = "C:/d_drive/projects/Project1/research/literature"
files = [f for f in os.listdir(lit_dir) if f.endswith('.json') and not f.startswith('_') and not f.startswith('__')]
papers = []

for f in files:
    try:
        with open(os.path.join(lit_dir, f), 'r', encoding='utf-8') as file:
            data = json.load(file)
            papers.append({
                "title": data.get("title", ""),
                "venue": data.get("venue", "Journal"),
                "year": data.get("year", "202x"),
                "abstract": data.get("abstract", "")[:50] + "..." # Just a short finding
            })
            if len(papers) == 15:
                break
    except:
        pass

with open("C:/d_drive/projects/Project1/papers_15.json", "w") as f:
    json.dump(papers, f, indent=2)
