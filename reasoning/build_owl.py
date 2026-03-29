# reasoning/build_owl.py
from rdflib import Graph, Namespace, URIRef, Literal

EX = Namespace("http://example.org/auto#")

def triples_to_owl(triples, output_path="outputs/generated.owl"):
    g = Graph()
    g.bind("ex", EX)
    for s, p, o in triples:
        g.add((URIRef(EX + s.replace(" ", "_")),
               URIRef(EX + p.replace(" ", "_")),
               URIRef(EX + o.replace(" ", "_"))))
    g.serialize(destination=output_path, format="xml")
    return output_path
