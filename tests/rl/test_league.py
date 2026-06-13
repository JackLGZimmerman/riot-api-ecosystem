import numpy as np

from app.rl.league import League, LeagueEntry, elo_update, sprt


def test_sprt_three_states():
    assert sprt(0, 0, 0) == "continue"
    assert sprt(300, 100, 0) == "accept"
    assert sprt(100, 300, 0) == "reject"
    assert sprt(5, 5, 0) == "continue"


def test_elo_conserves_and_directs():
    a, b = elo_update(1500.0, 1500.0, 1.0)
    assert a > 1500.0 > b
    assert abs((a + b) - 3000.0) < 1e-9
    a2, b2 = elo_update(1500.0, 1500.0, 0.5)
    assert abs(a2 - 1500.0) < 1e-9 and abs(b2 - 1500.0) < 1e-9


def test_pfsp_prefers_hard_opponents():
    rng = np.random.default_rng(0)
    lg = League(entries=[
        LeagueEntry(path="easy", games=100, wins=95),
        LeagueEntry(path="hard", games=100, wins=20),
    ])
    counts = [0, 0]
    for _ in range(2000):
        idx, _ = lg.sample_opponent(rng)
        counts[idx] += 1
    assert counts[1] > counts[0]


def test_admit_record_roundtrip(tmp_path):
    lg = League()
    e = lg.admit(b"DUMMYWEIGHTS", rating=1500.0, directory=tmp_path)
    assert lg.champion_idx == 0 and len(lg.entries) == 1
    assert (tmp_path / "entry_0.pt").read_bytes() == b"DUMMYWEIGHTS"
    lg.record(0, agent_won=True)
    lg.save(tmp_path)
    lg2 = League.load(tmp_path)
    assert len(lg2.entries) == 1 and lg2.entries[0].wins == 1
    assert lg2.entries[0].path == e.path
    assert lg2.champion is not None and lg2.champion.path == e.path
