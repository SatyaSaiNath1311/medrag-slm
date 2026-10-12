import csv
P = "outputs/analysis_nli/label_sheet.csv"
col = "your_label_1_supports_0_not"
rows = list(csv.DictReader(open(P, encoding="utf-8-sig")))
cols = list(rows[0].keys())

def save():
    with open(P, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

todo = [r for r in rows if not r[col].strip()]
done = len(rows) - len(todo)
print(f"{done}/{len(rows)} already labeled")
for i, r in enumerate(todo, 1):
    print("\n" + "=" * 90)
    print(f"[{done + i}/{len(rows)}] QUESTION: {r['question']}")
    print(f"\nCHOSEN ANSWER: {r['chosen_answer']}")
    print(f"\nPASSAGE:\n{r['passage']}")
    while True:
        a = input("\nDoes the passage state or clearly imply the chosen answer?  1=yes  0=no  q=quit > ").strip().lower()
        if a in ("0", "1"):
            r[col] = a; save(); break
        if a == "q":
            save(); raise SystemExit("saved; rerun to continue")
        print("enter 1, 0 or q")
print("\nall 50 labeled -> now run: python3 nli_agreement.py")
