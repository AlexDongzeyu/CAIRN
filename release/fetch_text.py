"""Reconstruct segment text from Densho given the released identifiers.

Usage:  python fetch_text.py segments_standoff.csv out.jsonl
Crawls politely at 1 request/second with an identifying User-Agent.
"""
import csv, json, sys, time, urllib.request

UA = "CAIRN-replication/0.1 (contact: research@example.org)"

def fetch(seg_id):
    req = urllib.request.Request(f"https://ddr.densho.org/api/0.2/{seg_id}/",
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)

def main(index_csv, out_path):
    with open(index_csv, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as o:
        for row in csv.DictReader(f):
            time.sleep(1.0)
            try:
                d = fetch(row["segment_id"])
            except Exception as e:
                print("skip", row["segment_id"], e); continue
            o.write(json.dumps({"segment_id": row["segment_id"],
                                "title": d.get("title"),
                                "description": d.get("description")}) + "\n")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
