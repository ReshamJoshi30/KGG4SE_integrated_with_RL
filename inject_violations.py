"""
inject_violations.py
====================
Inject a controlled set of challenging OWL violations into an existing
knowledge graph, producing a "broken" KG that the RL repair agent must fix.

Why this matters
----------------
The RL agent's real KG (merged_kg.owl) only has 3 disjoint violations of
the same type.  This means:
  - The agent sees the same state vector every episode
  - One action always fixes everything
  - There is nothing meaningful to learn

This script adds 8 additional violations of DIFFERENT types to the KG,
creating a challenging multi-violation repair task:

  V1  disjoint_violation           (+2 new — different entity pairs)
  V2  transitive_disjoint_violation (+1 new — via subclass chain)
  V3  class_used_as_individual     (+1 new)
  V4  functional_property_violation (+1 new)
  V5  range_violation               (+1 new)
  V6  external_type_violation       (+1 new)
  V7  unsat_class (double disjoint) (+1 new)

Together these produce 8-12 repair candidates across 6 distinct error
types, forcing the agent to learn type-specific repair strategies.

Usage
-----
# Inject into your real merged_kg.owl (creates a backup first):
python inject_violations.py

# Inject into a specific file:
python inject_violations.py --input outputs/intermediate/merged_kg.owl \
                             --output outputs/intermediate/merged_kg_broken.owl

# Use the standalone benchmark KG (no base KG needed):
python inject_violations.py --standalone
"""

import argparse
import shutil
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# The violation block to inject — raw OWL/XML string
# ---------------------------------------------------------------------------

VIOLATION_BLOCK = '''
  <!-- ================================================================ -->
  <!-- INJECTED VIOLATIONS — inserted by inject_violations.py           -->
  <!-- Date: {TIMESTAMP}                                                 -->
  <!-- These violations were deliberately added to benchmark RL repair.  -->
  <!-- ================================================================ -->

  <!-- === NEW CLASSES FOR VIOLATION CONTEXT === -->

  <owl:Class rdf:about="http://cpsagila.cs.uni-kl.de/GENIALOnt#AnalogComponent">
    <rdfs:label>AnalogComponent</rdfs:label>
    <rdfs:subClassOf rdf:resource="http://cpsagila.cs.uni-kl.de/GENIALOnt#ElectricalComponent"/>
    <owl:disjointWith rdf:resource="http://cpsagila.cs.uni-kl.de/GENIALOnt#DigitalComponent"/>
  </owl:Class>

  <owl:Class rdf:about="http://cpsagila.cs.uni-kl.de/GENIALOnt#DigitalComponent">
    <rdfs:label>DigitalComponent</rdfs:label>
    <rdfs:subClassOf rdf:resource="http://cpsagila.cs.uni-kl.de/GENIALOnt#ElectricalComponent"/>
  </owl:Class>

  <owl:Class rdf:about="http://cpsagila.cs.uni-kl.de/GENIALOnt#Signal">
    <rdfs:label>Signal</rdfs:label>
  </owl:Class>

  <!-- hasMeasuredValue: functional property (max one value per individual) -->
  <owl:DatatypeProperty rdf:about="http://cpsagila.cs.uni-kl.de/GENIALOnt#hasMeasuredValue">
    <rdfs:label>hasMeasuredValue</rdfs:label>
    <rdf:type rdf:resource="http://www.w3.org/2002/07/owl#FunctionalProperty"/>
    <rdfs:domain rdf:resource="http://cpsagila.cs.uni-kl.de/GENIALOnt#Sensor"/>
    <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#float"/>
  </owl:DatatypeProperty>

  <owl:ObjectProperty rdf:about="http://cpsagila.cs.uni-kl.de/GENIALOnt#transmitsSignal">
    <rdfs:label>transmitsSignal</rdfs:label>
    <rdfs:domain rdf:resource="http://cpsagila.cs.uni-kl.de/GENIALOnt#CommunicationBus"/>
    <rdfs:range rdf:resource="http://cpsagila.cs.uni-kl.de/GENIALOnt#Signal"/>
  </owl:ObjectProperty>

  <!-- === V1: DISJOINT VIOLATION — TempSensor typed as Actuator === -->
  <!-- TemperatureSensor subclasses Sensor; Sensor disjointWith Actuator -->
  <owl:NamedIndividual rdf:about="http://cpsagila.cs.uni-kl.de/GENIALOnt/ind/INJ_TempSensor_V1">
    <rdf:type rdf:resource="http://cpsagila.cs.uni-kl.de/GENIALOnt#TemperatureSensor"/>
    <rdf:type rdf:resource="http://cpsagila.cs.uni-kl.de/GENIALOnt#Actuator"/>
    <rdfs:label>INJ_TempSensor_V1</rdfs:label>
    <rdfs:comment>INJECTED: disjoint_violation — Sensor vs Actuator</rdfs:comment>
    <ont:hasSerialNumber xmlns:ont="http://cpsagila.cs.uni-kl.de/GENIALOnt#">SN-INJ-V1-001</ont:hasSerialNumber>
  </owl:NamedIndividual>

  <!-- === V2: DISJOINT VIOLATION — PressureSensor also typed as Actuator === -->
  <owl:NamedIndividual rdf:about="http://cpsagila.cs.uni-kl.de/GENIALOnt/ind/INJ_PressureSensor_V2">
    <rdf:type rdf:resource="http://cpsagila.cs.uni-kl.de/GENIALOnt#PressureSensor"/>
    <rdf:type rdf:resource="http://cpsagila.cs.uni-kl.de/GENIALOnt#Actuator"/>
    <rdfs:label>INJ_PressureSensor_V2</rdfs:label>
    <rdfs:comment>INJECTED: disjoint_violation — different entity, same type conflict</rdfs:comment>
    <ont:hasSerialNumber xmlns:ont="http://cpsagila.cs.uni-kl.de/GENIALOnt#">SN-INJ-V2-001</ont:hasSerialNumber>
  </owl:NamedIndividual>

  <!-- === V3: TRANSITIVE DISJOINT — Controller (DigitalComponent) + AnalogComponent === -->
  <!-- Controller subclasses DigitalComponent; DigitalComponent disjointWith AnalogComponent -->
  <owl:NamedIndividual rdf:about="http://cpsagila.cs.uni-kl.de/GENIALOnt/ind/INJ_SpeedCtrl_V3">
    <rdf:type rdf:resource="http://cpsagila.cs.uni-kl.de/GENIALOnt#Controller"/>
    <rdf:type rdf:resource="http://cpsagila.cs.uni-kl.de/GENIALOnt#AnalogComponent"/>
    <rdfs:label>INJ_SpeedCtrl_V3</rdfs:label>
    <rdfs:comment>INJECTED: transitive_disjoint_violation — Controller(Digital) vs AnalogComponent</rdfs:comment>
    <ont:hasSerialNumber xmlns:ont="http://cpsagila.cs.uni-kl.de/GENIALOnt#">SN-INJ-V3-001</ont:hasSerialNumber>
  </owl:NamedIndividual>

  <!-- === V4: FUNCTIONAL PROPERTY VIOLATION — two values for hasMeasuredValue === -->
  <owl:NamedIndividual rdf:about="http://cpsagila.cs.uni-kl.de/GENIALOnt/ind/INJ_WireSensor_V4">
    <rdf:type rdf:resource="http://cpsagila.cs.uni-kl.de/GENIALOnt#Sensor"/>
    <rdfs:label>INJ_WireSensor_V4</rdfs:label>
    <rdfs:comment>INJECTED: functional_property_violation — two hasMeasuredValue values</rdfs:comment>
    <ont:hasMeasuredValue xmlns:ont="http://cpsagila.cs.uni-kl.de/GENIALOnt#"
        rdf:datatype="http://www.w3.org/2001/XMLSchema#float">12.0</ont:hasMeasuredValue>
    <ont:hasMeasuredValue xmlns:ont="http://cpsagila.cs.uni-kl.de/GENIALOnt#"
        rdf:datatype="http://www.w3.org/2001/XMLSchema#float">24.0</ont:hasMeasuredValue>
  </owl:NamedIndividual>

  <!-- === V5: RANGE VIOLATION — transmitsSignal with a Literal instead of Signal individual === -->
  <owl:NamedIndividual rdf:about="http://cpsagila.cs.uni-kl.de/GENIALOnt/ind/INJ_CANBus_V5">
    <rdf:type rdf:resource="http://cpsagila.cs.uni-kl.de/GENIALOnt#CommunicationBus"/>
    <rdfs:label>INJ_CANBus_V5</rdfs:label>
    <rdfs:comment>INJECTED: range_violation — transmitsSignal with Literal value</rdfs:comment>
    <ont:transmitsSignal xmlns:ont="http://cpsagila.cs.uni-kl.de/GENIALOnt#"
        rdf:datatype="http://www.w3.org/2001/XMLSchema#string">engine_rpm_raw_string</ont:transmitsSignal>
  </owl:NamedIndividual>

  <!-- === V6: EXTERNAL TYPE VIOLATION — typed as schema:Device === -->
  <owl:NamedIndividual rdf:about="http://cpsagila.cs.uni-kl.de/GENIALOnt/ind/INJ_BrakeSensor_V6">
    <rdf:type rdf:resource="http://cpsagila.cs.uni-kl.de/GENIALOnt#Sensor"/>
    <rdf:type rdf:resource="http://schema.org/Device"/>
    <rdfs:label>INJ_BrakeSensor_V6</rdfs:label>
    <rdfs:comment>INJECTED: external_type_violation — schema:Device on a local Sensor</rdfs:comment>
    <ont:hasMeasuredValue xmlns:ont="http://cpsagila.cs.uni-kl.de/GENIALOnt#"
        rdf:datatype="http://www.w3.org/2001/XMLSchema#float">0.85</ont:hasMeasuredValue>
  </owl:NamedIndividual>

  <!-- === V7: UNSAT / DOUBLE DISJOINT TYPE === -->
  <!-- TemperatureSensor disjointWith PressureSensor -->
  <!-- Typing an individual as both makes it logically unsatisfiable -->
  <owl:NamedIndividual rdf:about="http://cpsagila.cs.uni-kl.de/GENIALOnt/ind/INJ_HybridSensor_V7">
    <rdf:type rdf:resource="http://cpsagila.cs.uni-kl.de/GENIALOnt#TemperatureSensor"/>
    <rdf:type rdf:resource="http://cpsagila.cs.uni-kl.de/GENIALOnt#PressureSensor"/>
    <rdfs:label>INJ_HybridSensor_V7</rdfs:label>
    <rdfs:comment>INJECTED: unsat_class — typed as both TemperatureSensor and PressureSensor (disjoint)</rdfs:comment>
    <ont:hasMeasuredValue xmlns:ont="http://cpsagila.cs.uni-kl.de/GENIALOnt#"
        rdf:datatype="http://www.w3.org/2001/XMLSchema#float">55.0</ont:hasMeasuredValue>
  </owl:NamedIndividual>

  <!-- === V8: CLASS USED AS INDIVIDUAL — Sensor class has a property assertion === -->
  <rdf:Description rdf:about="http://cpsagila.cs.uni-kl.de/GENIALOnt#Sensor">
    <ont:hasSerialNumber xmlns:ont="http://cpsagila.cs.uni-kl.de/GENIALOnt#">CLASS-WRONGLY-ASSERTED</ont:hasSerialNumber>
    <rdfs:comment>INJECTED: class_used_as_individual — Sensor class has property assertion</rdfs:comment>
  </rdf:Description>

  <!-- ================================================================ -->
  <!-- END OF INJECTED VIOLATIONS                                        -->
  <!-- ================================================================ -->
'''


def inject_into_owl(input_path: Path, output_path: Path) -> None:
    """
    Inject violation triples into an existing OWL file by inserting
    the violation XML block just before the closing </rdf:RDF> tag.

    Args:
        input_path:  Path to source OWL file.
        output_path: Path to write the broken OWL file.
    """
    content = input_path.read_text(encoding="utf-8")

    # Ensure it's an RDF/XML file
    if "</rdf:RDF>" not in content:
        raise ValueError(
            f"{input_path} does not appear to be RDF/XML (no </rdf:RDF> closing tag). "
            "Only RDF/XML format is supported by this injector."
        )

    # Add required namespace declarations if missing
    ns_additions = []
    if 'xmlns:schema=' not in content:
        ns_additions.append('xmlns:schema="http://schema.org/"')

    # Insert namespace into opening rdf:RDF tag if needed
    if ns_additions:
        for ns in ns_additions:
            # Find the rdf:RDF opening tag and inject before the >
            import re
            content = re.sub(
                r'(<rdf:RDF\b[^>]*?)(\s*>)',
                lambda m: m.group(1) + f'\n    {ns}' + m.group(2),
                content,
                count=1,
            )

    block = VIOLATION_BLOCK.replace("{TIMESTAMP}", datetime.now().isoformat())
    content = content.replace("</rdf:RDF>", block + "\n</rdf:RDF>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(f"[inject_violations] Written: {output_path}")
    print(f"[inject_violations] Violations injected:")
    print(f"  V1 — disjoint_violation         (INJ_TempSensor_V1: TemperatureSensor + Actuator)")
    print(f"  V2 — disjoint_violation         (INJ_PressureSensor_V2: PressureSensor + Actuator)")
    print(f"  V3 — transitive_disjoint        (INJ_SpeedCtrl_V3: Controller→Digital + Analog)")
    print(f"  V4 — functional_property        (INJ_WireSensor_V4: two hasMeasuredValue values)")
    print(f"  V5 — range_violation            (INJ_CANBus_V5: transmitsSignal-> Literal)")
    print(f"  V6 — external_type_violation    (INJ_BrakeSensor_V6: schema:Device type)")
    print(f"  V7 — unsat_class                (INJ_HybridSensor_V7: Temp + Pressure disjoint)")
    print(f"  V8 — class_used_as_individual   (Sensor class has property assertion)")
    print()
    print(f"[inject_violations] Run the reasoner to verify violations are detected:")
    print(f"  python -m rl.train_repair --no-human --owl {output_path} --episodes 100 --max-steps 20")


def main():
    import config

    parser = argparse.ArgumentParser(
        description="Inject OWL violations for RL repair benchmarking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path,
        default=config.DEFAULT_PATHS["repair_kg"]["input"],
        help="Base OWL file to inject violations into (your real merged_kg.owl).",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/intermediate/merged_kg_broken.owl"),
        help="Output path for the broken OWL file.",
    )
    parser.add_argument(
        "--standalone", action="store_true",
        help=(
            "Use the bundled benchmark_kg_challenging.owl instead of your real KG. "
            "Useful if merged_kg.owl does not exist yet."
        ),
    )
    parser.add_argument(
        "--backup", action="store_true", default=True,
        help="Create a timestamped backup of the input file before injection.",
    )
    args = parser.parse_args()

    if args.standalone:
        # Use the bundled benchmark KG
        standalone_path = Path(__file__).parent / "benchmark_kg_challenging.owl"
        if not standalone_path.exists():
            print(f"[ERROR] Standalone benchmark KG not found: {standalone_path}")
            print("  Make sure benchmark_kg_challenging.owl is in the same directory.")
            sys.exit(1)
        print(f"[inject_violations] Using standalone benchmark KG: {standalone_path}")
        output = args.output if args.output != Path("outputs/intermediate/merged_kg_broken.owl") \
                 else Path("outputs/intermediate/benchmark_kg_broken.owl")
        shutil.copy(standalone_path, output)
        print(f"[inject_violations] Benchmark KG already contains all violations.")
        print(f"[inject_violations] Copied to: {output}")
        print()
        print(f"Run RL training on it:")
        print(f"  python -m rl.train_repair --no-human --owl {output} --episodes 100 --max-steps 20")
        return

    if not args.input.exists():
        print(f"[ERROR] Input OWL not found: {args.input}")
        print("  Run 'python pipeline.py build-kg' first, or use --standalone.")
        sys.exit(1)

    # Backup
    if args.backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = args.input.parent / f"{args.input.stem}_backup_{ts}{args.input.suffix}"
        shutil.copy(args.input, backup)
        print(f"[inject_violations] Backup created: {backup}")

    inject_into_owl(args.input, args.output)


if __name__ == "__main__":
    main()