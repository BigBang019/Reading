from scholarly import scholarly
import bibtexparser
import json
from tqdm import tqdm
from collections import defaultdict

if __name__ == "__main__":
    search_query = scholarly.search_pubs('Draco Compression')
    conf_cnt = defaultdict(lambda:0)
    for pub in tqdm(search_query):
        bib_str = scholarly.bibtex(pub)
        bib = bibtexparser.loads(bib_str).entries[0]
        conf = None
        for key in ['booktitle', 'journal']:
            if key in bib:
                conf = bib[key]
        if conf is None:
            print(bib)
            break
        conf_cnt[conf] += 1
    with open('conf_cnt.json', 'w') as f:
        json.dump(conf_cnt, f)