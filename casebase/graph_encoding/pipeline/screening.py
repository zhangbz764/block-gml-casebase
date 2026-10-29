"""Explicit initial screening; SQL is accessed only with --database, never on import/help."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pandas as pd
from .screen_blocks import METRIC_FIELDS,RULES,screen_dataframe
PIPELINE_DIR=Path(__file__).resolve().parent

def stable_csv_bytes(frame: pd.DataFrame) -> bytes:
    text = frame.to_csv(index=False, lineterminator="\n", float_format="%.15g")
    return b"\xef\xbb\xbf" + text.encode("utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stable_csv_bytes(frame))


def _write_gzip_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(stable_csv_bytes(frame))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_database_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for field in METRIC_FIELDS:
        result[field] = pd.to_numeric(result[field], errors="coerce")
    for field in ("geom_is_null", "geom_is_empty", "geom_is_valid", "geom_area_positive"):
        result[field] = result[field].astype(bool)
    return result


def _fetch_candidate_frame(connection) -> pd.DataFrame:
    query = (PIPELINE_DIR / "query_candidate_blocks.sql").read_text(encoding="utf-8")
    with connection.cursor() as cursor:
        cursor.execute(query)
        columns = [item.name for item in cursor.description]
        chunks = []
        while True:
            rows = cursor.fetchmany(25_000)
            if not rows:
                break
            chunks.append(pd.DataFrame.from_records(rows, columns=columns))
    if not chunks:
        raise RuntimeError("Candidate query returned no rows")
    return _normalize_database_frame(pd.concat(chunks, ignore_index=True))



def main():
    p=argparse.ArgumentParser(description=__doc__)
    choice=p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--input',type=Path,help='candidate CSV or CSV.GZ with the original schema')
    choice.add_argument('--database',action='store_true',help='explicit read-only candidate query')
    p.add_argument('--output',type=Path,required=True,help='new output directory, never overwrites')
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('output already exists: '+str(args.output))
    if args.database:
        from .db_readonly import readonly_connection
        with readonly_connection(statement_timeout_ms=600000) as connection: frame=_fetch_candidate_frame(connection)
    else:
        frame=pd.read_csv(args.input,encoding='utf-8-sig',float_precision='round_trip',low_memory=False)
    ledger=screen_dataframe(frame)
    args.output.mkdir(parents=True,exist_ok=False)
    _write_gzip_csv(args.output/'block_screening_ledger.csv.gz',ledger)
    summary={'input_rows':len(ledger),'cities':int(ledger['city'].nunique()),
             'rule_version':RULES['rule_version'],'thresholds':ledger.attrs['thresholds'],
             'ledger_sha256':_sha256(args.output/'block_screening_ledger.csv.gz')}
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))
    return 0

if __name__=='__main__': raise SystemExit(main())
