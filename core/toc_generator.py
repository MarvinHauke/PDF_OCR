def generate_toc(entries):
    seen = set()
    toc = []
    for entry in entries:
        key = (entry["page"], entry["title"])
        if key not in seen:
            seen.add(key)
            toc.append(entry)
    return toc
