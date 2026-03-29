"""
alignment/ontology_index.py

Build a simple lexical index over ontology class names (and optionally
rdfs:label annotations later). Used to suggest remapping targets in QA repair.
"""
import re
from pathlib import Path
from owlready2 import get_ontology

def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower())

class OntologyIndex:
    def __init__(self, owl_path: str):
        p = Path(owl_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Ontology not found: {p}")
        # self.onto = get_ontology(p.as_uri()).load()
        self.onto = get_ontology(str(p)).load()
        self.label2iri = {}
        for cls in self.onto.classes():
            n = _norm(cls.name)
            if n and n not in self.label2iri:
                self.label2iri[n] = cls.iri
        self.labels = list(self.label2iri.keys())

    def _sim(self, a: str, b: str) -> float:
        # character overlap Jaccard (cheap; replace with embeddings later)
        sa, sb = set(a), set(b)
        return len(sa & sb) / max(1, len(sa | sb))

    def suggest(self, label: str, topn: int = 3):
        n = _norm(label)
        scores = [(self._sim(n, lab), lab) for lab in self.labels]
        scores.sort(reverse=True)
        return [self.label2iri[lab] for _, lab in scores[:topn]]
