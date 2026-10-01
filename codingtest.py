import re
from typing import Dict, Tuple
import requests
from bs4 import BeautifulSoup


def print_secret_grid(google_doc_url: str) -> None:
    html = requests.get(google_doc_url, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")

    # Collect (x, y) -> char
    points: Dict[Tuple[int, int], str] = {}

    for tr in soup.find_all("tr"):
        tds = tr.find_all(["td", "th"])
        if len(tds) < 3:
            continue

        x_txt = tds[0].get_text(strip=True)
        ch_txt = tds[1].get_text(strip=True)
        y_txt = tds[2].get_text(strip=True)

        # Only accept rows that look like: int, (1 char), int
        if not re.fullmatch(r"\d+", x_txt):
            continue
        if not re.fullmatch(r"\d+", y_txt):
            continue
        if ch_txt == "":
            continue

        x = int(x_txt)
        y = int(y_txt)
        # If the cell contains multiple characters (unlikely), keep it as-is.
        points[(x, y)] = ch_txt

    if not points:
        raise ValueError("No coordinate data found. Is the URL a published Google Doc with the expected table format?")

    max_x = max(x for x, _ in points.keys())
    max_y = max(y for _, y in points.keys())

    # Build grid (y increases downward, like the example)
    grid = [[" "] * (max_x + 1) for _ in range(max_y + 1)]
    for (x, y), ch in points.items():
        grid[y][x] = ch

    for row in grid:
        print("".join(row))


# Example usage:
print_secret_grid("https://docs.google.com/document/d/e/2PACX-1vRPzbNQcx5UriHSbZ-9vmsTow_R6RRe7eyAU60xIF9Dlz-vaHiHNO2TKgDi7jy4ZpTpNqM7EvEcfr_p/pub")
