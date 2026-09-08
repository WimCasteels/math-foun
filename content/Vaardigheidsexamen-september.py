"""Genereer een Moodle GIFT-importbestand uit de drie theorie-examenvragen
markdown-bestanden (algebra, analyse, statistiek).

Gebruik:  python theorie_examenvragen_generate.py
Output:   theorie_examenvragen_moodle.txt   (importeren via Moodle > Vragenbank
          > Importeren > GIFT-formaat)
"""

import re
from pathlib import Path

HERE = Path(__file__).parent

SOURCES = [
    ("Algebra", HERE / "algebra" / "theorie_examenvragen_algebra.md"),
    ("Analyse", HERE / "analyse" / "theorie_examenvragen_analyse.md"),
    ("Statistiek", HERE / "statistiek" / "theorie_examenvragen_statistiek.md"),
]


def clean(text: str) -> str:
    """Verwijder markdown-opmaak (bold, inline code) uit gewone tekst."""
    text = text.replace("**", "")
    text = text.replace("`", "")
    text = re.sub(r"\*\((.*?)\)\*", r"(\1)", text)  # *(tip)* -> (tip)
    return text.strip()


def gift_escape(text: str) -> str:
    """Escape de GIFT-besturingstekens (: blijft veilig in bodytekst)."""
    text = text.replace("\\", "\\\\")
    for ch in ["~", "=", "#", "{", "}"]:
        text = text.replace(ch, "\\" + ch)
    return text


def parse_blocks(md: str):
    """Splits de markdown op in vraagblokken (negeert de antwoordsleutel)."""
    parts = re.split(r"^## ", md, flags=re.MULTILINE)
    for part in parts:
        if part.startswith("Vraag "):
            yield part


def parse_question(block: str):
    lines = block.splitlines()
    heading = lines[0]
    title = heading.split("—", 1)[1].strip() if "—" in heading else heading.strip()

    qtype = None
    for ln in lines:
        if "**Type:**" in ln:
            low = ln.lower()
            if "multichoice" in low:
                qtype = "multichoice"
            elif "numerical" in low:
                qtype = "numerical"
            elif "shortanswer" in low:
                qtype = "shortanswer"
            break

    # vraagtekst = prozaregels (geen heading/type/tabel/bullets/scheidingslijn)
    prose = []
    for ln in lines[1:]:
        s = ln.strip()
        if not s or s == "---":
            continue
        if s.startswith("##") or s.startswith("|") or s.startswith("- "):
            continue
        if s.startswith(">") or "**Type:**" in s:
            continue
        prose.append(clean(s))
    qtext = " ".join(prose).strip()

    options, answers, num_answer, num_tol = [], [], None, None

    if qtype == "multichoice":
        for ln in lines:
            s = ln.strip()
            if not s.startswith("|"):
                continue
            cells = [c.strip() for c in s.strip("|").split("|")]
            if len(cells) < 2:
                continue
            if cells[0].lower().startswith("antwoordoptie") or set(cells[0]) <= {"-", ":"}:
                continue
            opt = clean(cells[0])
            frac_digits = re.sub(r"[^0-9]", "", cells[1])
            correct = frac_digits == "100"
            options.append((opt, correct))

    elif qtype == "numerical":
        for ln in lines:
            s = ln.strip()
            m = re.match(r"- Juist antwoord:\s*\*\*(.+?)\*\*", s)
            if m:
                num_answer = m.group(1).strip()
            m = re.match(r"- Tolerantie:\s*\*\*(.+?)\*\*", s)
            if m:
                num_tol = m.group(1).strip()

    elif qtype == "shortanswer":
        for ln in lines:
            s = ln.strip()
            if s.startswith("- Aanvaarde antwoorden:"):
                answers = re.findall(r"`([^`]+)`", s)

    return {
        "title": title,
        "type": qtype,
        "text": qtext,
        "options": options,
        "answers": answers,
        "num_answer": num_answer,
        "num_tol": num_tol,
    }


def to_gift(q: dict, number: int) -> str:
    title = gift_escape(q["title"])
    text = gift_escape(q["text"])
    head = f"::{number:02d}. {title}:: {text} "

    if q["type"] == "multichoice":
        body = ["{"]
        for opt, correct in q["options"]:
            prefix = "=" if correct else "~"
            body.append(f"{prefix}{gift_escape(opt)}")
        body.append("}")
        return head + "\n".join(body)

    if q["type"] == "numerical":
        tol = q["num_tol"] if q["num_tol"] not in (None, "") else "0"
        return head + f"{{#{q['num_answer']}:{tol}}}"

    if q["type"] == "shortanswer":
        body = ["{"] + [f"={gift_escape(a)}" for a in q["answers"]] + ["}"]
        return head + "\n".join(body)

    return head + "{}"


def main():
    out = ["// Moodle GIFT-import — theorie-examenvragen",
           "// Importeren via: Vragenbank > Importeren > GIFT-formaat (UTF-8)",
           ""]
    total = 0
    for subject, path in SOURCES:
        md = path.read_text(encoding="utf-8")
        out.append(f"$CATEGORY: $course$/Theorie-examen/{subject}")
        out.append("")
        for i, block in enumerate(parse_blocks(md), start=1):
            q = parse_question(block)
            if not q["type"]:
                continue
            out.append(to_gift(q, i))
            out.append("")
            total += 1
        out.append("")

    target = HERE / "theorie_examenvragen_moodle.txt"
    target.write_text("\n".join(out), encoding="utf-8")
    print(f"{total} vragen geschreven naar {target}")


if __name__ == "__main__":
    main()
