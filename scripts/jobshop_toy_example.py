"""The hand-sized example of the job-shop tutorial: one machine, four waiting orders, two colours.

The machine paints parts. It is set up for blue; switching colour (a changeover) takes 10 min.
Each rule picks the next order whenever the machine is free; the best order is found by
trying all 24 sequences. Prints the schedule and the lateness of every order for each method.
Same timing rule as the repository's simulator: an order starts after the changeover it needs.
"""
from itertools import permutations

SETUP = 10
START_COLOUR = "blue"
# name: (colour, processing time, due time, arrival order)
ORDERS = {
    "A": ("blue", 20, 30, 1),
    "B": ("red", 10, 25, 2),
    "C": ("blue", 15, 70, 3),
    "D": ("red", 25, 60, 4),
}


def run(sequence):
    t, colour, rows = 0, START_COLOUR, []
    for name in sequence:
        c, p, due, _ = ORDERS[name]
        setup = 0 if c == colour else SETUP
        start = t + setup
        end = start + p
        rows.append((name, t, start, end, max(0, end - due)))
        t, colour = end, c
    return rows


def rule_sequence(key):
    waiting, colour, seq = set(ORDERS), START_COLOUR, []
    while waiting:
        nxt = min(waiting, key=lambda n: key(n, colour))
        seq.append(nxt); waiting.remove(nxt); colour = ORDERS[nxt][0]
    return seq


RULES = {
    "FIFO (first come, first served)": lambda n, col: ORDERS[n][3],
    "SPT (shortest job first)": lambda n, col: (ORDERS[n][1], ORDERS[n][3]),
    "EDD (most urgent first)": lambda n, col: (ORDERS[n][2], ORDERS[n][3]),
    "SST (no colour change if possible)": lambda n, col: (0 if ORDERS[n][0] == col else SETUP, ORDERS[n][3]),
}

def all_results():
    results = {name: run(rule_sequence(key)) for name, key in RULES.items()}
    best = min(permutations(ORDERS), key=lambda s: (sum(r[4] for r in run(s)), "".join(s)))
    results["Best order (all 24 tried)"] = run(best)
    return results


def svg() -> str:
    """Inline SVG of the five schedules; colours follow the page's CSS variables (light and dark)."""
    results = all_results()
    names = {  # row label (plain words) and the abbreviation under it
        "FIFO (first come, first served)": ("First come, first served", "FIFO"),
        "SPT (shortest job first)": ("Shortest job first", "SPT"),
        "EDD (most urgent first)": ("Most urgent first", "EDD"),
        "SST (no colour change if possible)": ("Avoid colour changes", "SST"),
        "Best order (all 24 tried)": ("Best possible order", "all 24 orders tried"),
    }
    px, left, top, row, bar_h = 4.6, 200, 40, 62, 30
    width = left + 112 * px + 150
    fill = {"blue": "#2a78d6", "red": "#d9534f"}
    out = [f'<svg viewBox="0 0 {width:.0f} {top + row * len(results) + 10}" role="img" '
           'aria-labelledby="toyT toyD" class="diagram-svg">',
           '<title id="toyT">Five ways to order the same four paint jobs</title>',
           '<desc id="toyD">For each method, a timeline of the machine: grey striped blocks are colour '
           'changes, blue and red blocks are the orders A to D, and the total lateness is written on the right.</desc>',
           '<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
           '<rect width="6" height="6" style="fill:var(--surface)"/><line x1="0" y1="0" x2="0" y2="6" '
           'style="stroke:var(--muted);stroke-width:2.2"/></pattern></defs>']
    for t in range(0, 111, 10):
        x = left + t * px
        out.append(f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{top + row * len(results) - 18}" style="stroke:var(--rule)"/>')
        out.append(f'<text x="{x:.1f}" y="{top - 14}" text-anchor="middle" class="d-tick">{t}</text>')
    out.append(f'<text x="{left - 12}" y="{top - 14}" text-anchor="end" class="d-tick">minute</text>')
    for i, (name, rows) in enumerate(results.items()):
        y = top + i * row
        label, abbr = names[name]
        out.append(f'<text x="{left - 12}" y="{y + 14}" text-anchor="end" class="d-lbl">{label}</text>')
        out.append(f'<text x="{left - 12}" y="{y + 31}" text-anchor="end" class="d-sub">{abbr}</text>')
        for name_, free, start, end, late in rows:
            if start > free:
                out.append(f'<rect x="{left + free * px:.1f}" y="{y}" width="{(start - free) * px:.1f}" height="{bar_h}" fill="url(#hatch)"/>')
            out.append(f'<rect x="{left + start * px:.1f}" y="{y}" width="{(end - start) * px:.1f}" height="{bar_h}" rx="3" fill="{fill[ORDERS[name_][0]]}"/>')
            out.append(f'<text x="{left + (start + end) / 2 * px:.1f}" y="{y + 21}" text-anchor="middle" class="d-job">{name_}</text>')
            if late:
                out.append(f'<text x="{left + end * px - 2:.1f}" y="{y + bar_h + 15}" text-anchor="end" class="d-late">{late} min late</text>')
        total = sum(r[4] for r in rows)
        cls = "d-tot best" if name.startswith("Best") else "d-tot"
        out.append(f'<text x="{left + 114 * px:.1f}" y="{y + 14}" class="{cls}">{total} min</text>')
        out.append(f'<text x="{left + 114 * px:.1f}" y="{y + 31}" class="d-sub">late in total</text>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    results = all_results()
    for name, rows in results.items():
        total = sum(r[4] for r in rows)
        changes = sum(1 for r in rows if r[2] > r[1])
        print(f"{name:38s} order {''.join(r[0] for r in rows)}  total lateness {total:3d} min  "
              f"changeovers {changes}  finish {rows[-1][3]}  late: " +
              ", ".join(f"{r[0]} {r[4]}" for r in rows if r[4]))
