import fitz, sys, csv
files = ["data\dsa.pdf","data\java.pdf","data\leetcode.pdf","data\python.pdf","data\pythoncolab.pdf"]
rows=[]
for f in files:
    doc = fitz.open(f)
    total = sum(len(p.get_text("text")) for p in doc)
    pages = len(doc)
    zero_pages = sum(1 for p in doc if len(p.get_text("text"))==0)
    rows.append((f, pages, total, round(total/pages if pages else 0,2), zero_pages))
with open("pdf_report.csv","w",newline="") as out:
    writer=csv.writer(out)
    writer.writerow(["file","pages","total_chars","avg_chars_per_page","zero_text_pages"])
    writer.writerows(rows)
print("report written to pdf_report.csv")
