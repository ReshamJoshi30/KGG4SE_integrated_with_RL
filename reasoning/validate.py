# reasoning/validate.py
def load_ontology_text(ontology_path):
    with open(ontology_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def simple_validate(triples, ontology_text):
    valid = []
    rejected = []
    for t in triples:
        s, p, o = t
        # naive rule: keep if subject or object word appears in ontology file
        if s.lower() in ontology_text.lower() or o.lower() in ontology_text.lower():
            valid.append(t)
        else:
            rejected.append(t)
    return valid, rejected
