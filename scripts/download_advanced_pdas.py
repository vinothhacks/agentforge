"""Download public PDA-related documents into fixtures/advanced/source."""

from __future__ import annotations

from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "advanced" / "source"
ROOT.mkdir(parents=True, exist_ok=True)

FILES = [
    (
        "metacad-pda-proforma.html",
        "https://metacad.io/templates/pda-proforma.html",
    ),
    (
        "metacad-pda-proforma.csv",
        "https://metacad.io/templates/pda-proforma.csv",
    ),
    (
        "esteve-paldiski-brochure.pdf",
        "https://esteve.ee/wp-content/uploads/2023/03/esteve-brochure-2023-03-02.pdf",
    ),
    (
        "fonasba-standard-liner-agency-agreement-1993.pdf",
        "https://www.fonasba.com/wp-content/uploads/2012/02/STANDARD-LINER-AGENCY-AGREEMENT-1993.pdf",
    ),
    (
        "uscourts-pda-da-desk-opinion.pdf",
        "https://www.govinfo.gov/content/pkg/USCOURTS-txsd-4_11-cv-00938/pdf/USCOURTS-txsd-4_11-cv-00938-0.pdf",
    ),
    (
        "itic-intermediary-2006-proforma-da.pdf",
        "https://www.itic-insure.com/fileadmin/uploads/itic/Documents/Intermediary_2006.pdf",
    ),
]


def main() -> int:
    headers = {"User-Agent": "AgentForge-eval/0.1 (research corpus; local PDA-query tests)"}
    ok = 0
    with httpx.Client(timeout=60.0, follow_redirects=True, headers=headers) as client:
        for name, url in FILES:
            dest = ROOT / name
            try:
                r = client.get(url)
                r.raise_for_status()
                dest.write_bytes(r.content)
                print(f"OK {name} {len(r.content)} bytes from {url}")
                ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {name}: {exc}")
    (ROOT / "SOURCES.txt").write_text(
        "\n".join(f"{n}\t{u}" for n, u in FILES) + "\n",
        encoding="utf-8",
    )
    print(f"downloaded {ok}/{len(FILES)}")
    return 0 if ok >= 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
