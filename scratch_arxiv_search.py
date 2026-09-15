import urllib.request
import xml.etree.ElementTree as ET
import time

queries = [
    'all:federated+AND+all:recommendation',
    'all:popularity+AND+all:bias+AND+all:recommendation',
    'all:LLM+AND+all:recommendation'
]

results = []
for q in queries:
    url = f'http://export.arxiv.org/api/query?search_query={q}&sortBy=submittedDate&sortOrder=desc&max_results=50'
    try:
        req = urllib.request.urlopen(url)
        xml_data = req.read()
        root = ET.fromstring(xml_data)
        
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            published = entry.find('{http://www.w3.org/2005/Atom}published').text
            year = int(published[:4])
            if year in [2025, 2026]: 
                title = entry.find('{http://www.w3.org/2005/Atom}title').text.replace('\n', ' ')
                summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.replace('\n', ' ')
                authors = [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
                link = entry.find('{http://www.w3.org/2005/Atom}id').text
                results.append({'title': title, 'year': year, 'authors': authors, 'summary': summary, 'link': link})
    except Exception as e:
        print(f"Error querying {q}: {e}")
    time.sleep(1)

# Remove duplicates based on title
seen = set()
unique_results = []
for r in results:
    if r['title'] not in seen:
        seen.add(r['title'])
        unique_results.append(r)

unique_results.sort(key=lambda x: x['year'], reverse=True)

with open('c:/d_drive/projects/Project1/arxiv_recent.txt', 'w', encoding='utf-8') as f:
    for r in unique_results:
        f.write(f"Title: {r['title']}\n")
        f.write(f"Authors: {', '.join(r['authors'])}\n")
        f.write(f"Year: {r['year']}\n")
        f.write(f"URL: {r['link']}\n")
        f.write(f"Abstract: {r['summary'][:500]}...\n")
        f.write("-" * 80 + "\n")
print(f"Found {len(unique_results)} papers from 2025/2026.")
