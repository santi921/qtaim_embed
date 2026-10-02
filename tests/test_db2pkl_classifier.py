"""db2pkl_classifier: status filtering per DB mode, legacy flag semantics, dedup order."""

import sqlite3

import pandas as pd
import pytest

from qtaim_embed.scripts.helpers import db2pkl_classifier as d2p

WATER = "3\n\nO 0.000 0.000 0.117\nH 0.000 0.757 -0.469\nH 0.000 -0.757 -0.469\n"
METHANE = "5\n\nC 0 0 0\nH 0.63 0.63 0.63\nH -0.63 -0.63 0.63\nH -0.63 0.63 -0.63\nH 0.63 -0.63 -0.63\n"


def _make_db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE structures (id INTEGER, elements TEXT, natoms INTEGER, status TEXT, "
        "charge INTEGER, spin INTEGER, geometry TEXT, metal TEXT)"
    )
    conn.executemany("INSERT INTO structures VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def test_strict_mode_excludes_queued_rows(tmp_path):
    db = tmp_path / "a.db"
    _make_db(db, [
        (1, "OH", 3, "completed", 0, 1, WATER, "none"),
        (2, "CH", 5, "failed", 0, 1, METHANE, "none"),
        (3, "CH", 5, "to_run", 0, 1, METHANE.replace("0.63", "0.64"), "none"),
    ])
    df = d2p.convert_db_to_classifier_df([], [str(db)], num_workers=1)
    assert sorted(df["ids"].tolist()) == [1, 2]
    assert df.set_index("ids").loc[2, "success"] == 0.0


def test_extended_mode_labels_queued_as_failed(tmp_path):
    db = tmp_path / "a.db"
    _make_db(db, [
        (1, "OH", 3, "completed", 0, 1, WATER, "none"),
        (3, "CH", 5, "to_run", 0, 1, METHANE, "none"),
    ])
    df = d2p.convert_db_to_classifier_df([str(db)], [], num_workers=1)
    assert sorted(df["ids"].tolist()) == [1, 3]
    assert df.set_index("ids").loc[3, "success"] == 0.0


def test_completed_wins_dedup_over_nonterminal_copy(tmp_path):
    ext = tmp_path / "ext.db"
    strict = tmp_path / "strict.db"
    _make_db(ext, [(10, "OH", 3, "running", 0, 1, WATER, "none")])
    _make_db(strict, [(20, "OH", 3, "completed", 0, 1, WATER, "none")])
    df = d2p.convert_db_to_classifier_df([str(ext)], [str(strict)], num_workers=1)
    assert len(df) == 1
    assert df.iloc[0]["success"] == 1.0


def test_legacy_db_paths_flag_is_strict_and_requires_a_db(tmp_path):
    db = tmp_path / "a.db"
    _make_db(db, [
        (1, "OH", 3, "completed", 0, 1, WATER, "none"),
        (3, "CH", 5, "to_run", 0, 1, METHANE, "none"),
    ])
    out = tmp_path / "out.pkl"
    d2p.main(["--db_paths", str(db), "--output", str(out)])
    df = pd.read_pickle(out)
    assert df["ids"].tolist() == [1]
    with pytest.raises(SystemExit):
        d2p.main(["--output", str(out)])
