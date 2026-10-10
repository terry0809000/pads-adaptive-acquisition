"""Assemble the cumulative archive without altering any earlier-round entry."""

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-archive", type=Path, required=True)
    parser.add_argument("--staged-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    staged = args.staged_dir.resolve()
    output = args.output.resolve()
    assert output != args.previous_archive.resolve(), "Preserve the predecessor archive."
    assert not output.exists(), "Choose a new output path; never overwrite an archive."

    visual_qa = json.loads((source / "VISUAL_QA.json").read_text())
    integration_qa = json.loads((source / "INTEGRATION_VALIDATION.json").read_text())
    assert visual_qa["passed"] and integration_qa["passed"]
    staged_names = [
        "PADS_manuscript_v1.docx",
        "PADS_integrated_supplement.docx",
        "PADS_20_dimension_review_ZH.md",
    ]
    for name in staged_names:
        assert (staged / name).read_bytes() == (source / "integrated" / name).read_bytes(), name
    for document in visual_qa["documents"]:
        assert sha256((staged / document["document"]).read_bytes()) == document["sha256"]

    additions = {"00_START_HERE_ROUND6.md": staged / "00_START_HERE_ROUND6.md"}
    for path in sorted(source.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            additions["pads_analysis/reviewer_round6/" + path.relative_to(source).as_posix()] = path

    manifest = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "predecessor_archive": {
            "name": args.previous_archive.name,
            "sha256": sha256(args.previous_archive.read_bytes()),
        },
        "description": "Uncompressed entry hashes. Historical entries are retained unchanged; current documents are under reviewer_round6/integrated.",
        "entries": [],
    }
    with ZipFile(args.previous_archive) as previous, ZipFile(output, "w", ZIP_DEFLATED, compresslevel=6) as bundle:
        old_names = previous.namelist()
        assert len(old_names) == len(set(old_names)), "Duplicate predecessor paths."
        assert not set(old_names).intersection(additions), "New files would replace historical entries."
        manifest["predecessor_archive"]["entry_count"] = len(old_names)
        for info in previous.infolist():
            data = previous.read(info.filename)
            bundle.writestr(copy.copy(info), data)
            manifest["entries"].append({"path": info.filename, "bytes": len(data), "sha256": sha256(data), "origin": "round5_archive"})
        for name, path in additions.items():
            data = path.read_bytes()
            bundle.writestr(name, data)
            manifest["entries"].append({"path": name, "bytes": len(data), "sha256": sha256(data), "origin": "round6"})
        bundle.writestr("ROUND6_ARCHIVE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    # Verify the actual archive bytes, including every preserved historical entry.
    with ZipFile(output) as bundle:
        assert len(bundle.namelist()) == len(set(bundle.namelist()))
        assert bundle.testzip() is None
        for entry in manifest["entries"]:
            data = bundle.read(entry["path"])
            assert len(data) == entry["bytes"] and sha256(data) == entry["sha256"], entry["path"]
    report = {
        "passed": True,
        "historical_entries_verified": len(old_names),
        "new_entries_verified": len(additions),
        "staged_deliverables_equal_archived_documents": staged_names,
        "document_visual_qa_passed": True,
        "visually_checked_pages": sum(d["pages"] for d in visual_qa["documents"]),
        "integration_checks_passed": len(integration_qa["checks"]),
        "hash_scope": "Every historical and new source entry; the manifest and this report do not hash themselves.",
    }
    with ZipFile(output, "a", ZIP_DEFLATED, compresslevel=6) as bundle:
        bundle.writestr("ROUND6_PACKAGE_VALIDATION.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with ZipFile(output) as bundle:
        assert bundle.testzip() is None
        report["total_archive_entries"] = len(bundle.namelist())
    report["archive_sha256"] = sha256(output.read_bytes())
    report["archive_size_bytes"] = output.stat().st_size
    report["archive_path"] = str(output)
    (staged / "PADS_round6_package_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
