"""Fill in missing SMILES in drugs.csv by looking up drug names in PubChem.

    python datasets/scripts/fetch_smiles.py --drugs data/canonical/drugs.csv

Drugs PubChem cannot resolve keep an empty SMILES; the model then falls back to a
placeholder graph for them. Fix those by hand (or drop them) for best results.
"""
import argparse
import json
import time
import urllib.parse
import urllib.request

import pandas as pd

URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/{}/JSON"
KEYS = ("SMILES", "IsomericSMILES", "CanonicalSMILES", "ConnectivitySMILES")


def lookup(name):
    for prop in ("SMILES", "CanonicalSMILES"):
        try:
            url = URL.format(urllib.parse.quote(name), prop)
            with urllib.request.urlopen(url, timeout=20) as r:
                props = json.load(r)["PropertyTable"]["Properties"][0]
            for k in KEYS:
                if props.get(k):
                    return props[k]
        except Exception:
            continue
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drugs", default="data/canonical/drugs.csv")
    args = p.parse_args()
    drugs = pd.read_csv(args.drugs, dtype={"drug_id": str})
    drugs["smiles"] = drugs.get("smiles", pd.Series([""] * len(drugs))).fillna("")
    missing = drugs["smiles"] == ""
    for i in drugs.index[missing]:
        drugs.at[i, "smiles"] = lookup(drugs.at[i, "drug_name"])
        print("%-30s %s" % (drugs.at[i, "drug_name"], drugs.at[i, "smiles"][:60] or "NOT FOUND"))
        time.sleep(0.25)  # PubChem allows max 5 requests/second
    drugs.to_csv(args.drugs, index=False)
    print("resolved %d / %d" % ((drugs["smiles"] != "").sum(), len(drugs)))


if __name__ == "__main__":
    main()
