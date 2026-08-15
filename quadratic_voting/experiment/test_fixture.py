from __future__ import annotations

import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from quadratic_voting.experiment.runner import run_experiment
from quadratic_voting.experiment.store import acquire_writer_lock, open_sqlite_store
from quadratic_voting.experiment.test_runner import (
    FixedClock,
    ScriptedGenerator,
    make_runs,
)
from quadratic_voting.experiment.transcript import render_transcript
from quadratic_voting.experiment.types import (
    ElicitationArm,
    FinalResultEvent,
    PriorTurnEvent,
    RunStatus,
    TurnKind,
    VotingRegime,
)


class StageOneFixtureTests(unittest.TestCase):
    """GPU-free Stage-1 trajectory gate through the real store and renderer."""

    def test_six_runs_barrier_arm_order_terminal_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store, creation = make_runs(Path(directory) / "fixture.sqlite3")
            self.addCleanup(store.close)
            for (arm, _regime), run_id in creation.run_ids.items():
                self.assertIs(
                    run_experiment(
                        run_id,
                        store=store,
                        generator=ScriptedGenerator(),
                        clock=FixedClock(),
                    ),
                    RunStatus.COMPLETE,
                )
                before = store.connection.total_changes
                self.assertIs(
                    run_experiment(
                        run_id,
                        store=store,
                        generator=ScriptedGenerator(),
                        clock=FixedClock(),
                    ),
                    RunStatus.COMPLETE,
                )
                self.assertEqual(store.connection.total_changes, before)
                for _index, voter_id in store.voters(run_id):
                    view = store.voter_round_view(run_id, voter_id)
                    events = [
                        event.kind
                        for event in view.history
                        if isinstance(event, PriorTurnEvent) and event.round_index == 1
                    ]
                    expected = {
                        ElicitationArm.ACTION_ONLY: [TurnKind.BALLOT],
                        ElicitationArm.STATEMENT_THEN_ACTION: [
                            TurnKind.STATEMENT,
                            TurnKind.BALLOT,
                        ],
                        ElicitationArm.ACTION_THEN_STATEMENT: [
                            TurnKind.BALLOT,
                            TurnKind.STATEMENT,
                        ],
                    }[arm]
                    self.assertEqual(events, expected)
                    self.assertIsInstance(view.history[-1], FinalResultEvent)
                    rendered = render_transcript(view)
                    setup = rendered[0].content
                    positions = [
                        setup.index(f"[{candidate}]")
                        for candidate, _ in view.setup.candidate_cards
                    ]
                    self.assertEqual(positions, sorted(positions))
                    self.assertEqual(len(view.setup.candidate_cards), 3)
                    visible = "\n".join(
                        message.content for message in rendered
                    ).casefold()
                    for marker in (
                        "interrupted",
                        "resume marker",
                        "started_at",
                        "timestamp",
                    ):
                        self.assertNotIn(marker, visible)
                self.assertEqual(
                    store.connection.execute(
                        'SELECT COUNT(*) FROM "round" WHERE run_id=?', (run_id,)
                    ).fetchone()[0],
                    2,
                )

    def test_support_and_opposition_outcomes_obey_regime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store, creation = make_runs(Path(directory) / "fixture.sqlite3")
            self.addCleanup(store.close)
            for regime in VotingRegime:
                run_id = creation.run_ids[(ElicitationArm.ACTION_ONLY, regime)]
                run_experiment(
                    run_id,
                    store=store,
                    generator=ScriptedGenerator(),
                    clock=FixedClock(),
                )
                rows = store.connection.execute(
                    'SELECT o.protected_candidate_id,o.removed_candidate_id FROM "round" r '
                    "JOIN round_outcome o ON o.round_id=r.round_id WHERE r.run_id=?",
                    (run_id,),
                ).fetchall()
                self.assertTrue(rows)
                for protected, removed in rows:
                    if regime is VotingRegime.SUPPORT:
                        self.assertIsNotNone(protected)
                        self.assertNotEqual(protected, removed)
                    else:
                        self.assertIsNone(protected)

    def test_every_runner_commit_boundary_resumes_byte_identically(self) -> None:
        if not (os.name == "posix" and hasattr(signal, "SIGKILL")):
            self.skipTest("real transaction SIGKILL recovery requires POSIX signals")
        child_code = r"""
import sys
import time
from pathlib import Path
from quadratic_voting.experiment.runner import run_experiment
from quadratic_voting.experiment.store import open_sqlite_store
from quadratic_voting.experiment.test_runner import FixedClock, ScriptedGenerator
from quadratic_voting.experiment.types import RunId

db, raw_run_id, target_text = sys.argv[1:]
target = int(target_text)
def hook(ordinal: int) -> None:
    if ordinal == target:
        print(f"READY commit={ordinal}", flush=True)
        time.sleep(3600)
with open_sqlite_store(Path(db), commit_hook=hook) as store:
    run_experiment(RunId(raw_run_id), store=store, generator=ScriptedGenerator(), clock=FixedClock())
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed_db = root / "seed.sqlite3"
            seed_store, creation = make_runs(seed_db)
            run_id = creation.run_ids[
                (ElicitationArm.ACTION_ONLY, VotingRegime.OPPOSITION)
            ]
            seed_store.close()

            control_db = root / "control.sqlite3"
            shutil.copy2(seed_db, control_db)
            ordinals: list[int] = []
            control = open_sqlite_store(
                control_db, commit_hook=lambda ordinal: ordinals.append(ordinal)
            )
            run_experiment(
                run_id,
                store=control,
                generator=ScriptedGenerator(),
                clock=FixedClock(),
            )
            expected = tuple(
                tuple(
                    (message.role.value, message.content)
                    for message in render_transcript(
                        control.voter_round_view(run_id, voter_id)
                    )
                )
                for _index, voter_id in control.voters(run_id)
            )
            control.close()
            self.assertGreater(len(ordinals), 0)

            for kill_at in ordinals:
                crash_db = root / f"crash-{kill_at}.sqlite3"
                shutil.copy2(seed_db, crash_db)
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-c",
                        child_code,
                        str(crash_db),
                        str(run_id),
                        str(kill_at),
                    ],
                    cwd=Path(__file__).resolve().parents[2],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                try:
                    assert child.stdout is not None
                    ready, _, _ = select.select([child.stdout], [], [], 20)
                    if not ready:
                        assert child.stderr is not None
                        self.fail(
                            f"child did not reach transaction {kill_at}: "
                            f"{child.stderr.read()}"
                        )
                    self.assertEqual(
                        child.stdout.readline().strip(), f"READY commit={kill_at}"
                    )
                    os.kill(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
                    self.assertEqual(child.returncode, -signal.SIGKILL)
                finally:
                    if child.poll() is None:
                        child.kill()
                        child.wait(timeout=10)
                    if child.stdout is not None:
                        child.stdout.close()
                    if child.stderr is not None:
                        child.stderr.close()
                resumed = open_sqlite_store(crash_db)
                self.assertIs(
                    run_experiment(
                        run_id,
                        store=resumed,
                        generator=ScriptedGenerator(),
                        clock=FixedClock(),
                    ),
                    RunStatus.COMPLETE,
                )
                actual = tuple(
                    tuple(
                        (message.role.value, message.content)
                        for message in render_transcript(
                            resumed.voter_round_view(run_id, voter_id)
                        )
                    )
                    for _index, voter_id in resumed.voters(run_id)
                )
                self.assertEqual(actual, expected, f"commit boundary {kill_at}")
                duplicate = resumed.connection.execute(
                    "SELECT turn_id,attempt_index,COUNT(*) FROM model_call "
                    "WHERE status='committed' GROUP BY turn_id,attempt_index "
                    "HAVING COUNT(*) > 1"
                ).fetchall()
                self.assertEqual(duplicate, [])
                prompts = "\n".join(
                    row[0]
                    for row in resumed.connection.execute(
                        "SELECT prompt_messages_json FROM model_call"
                    )
                ).casefold()
                for marker in ("resume marker", "timestamp", "interruption"):
                    self.assertNotIn(marker, prompts)
                resumed.close()

    @unittest.skipUnless(
        os.name == "posix" and hasattr(signal, "SIGKILL"),
        "real runner SIGKILL recovery requires POSIX signals and flock",
    )
    def test_real_child_runner_sigkill_resumes_at_two_generation_points(self) -> None:
        try:
            import fcntl  # noqa: F401
        except ImportError:
            self.skipTest(
                "fcntl is unavailable, so the writer-lock contract cannot run"
            )

        child_code = r"""
import sys
import time
from collections.abc import Sequence

from llm_runtime.types import ChatMessage
from quadratic_voting.experiment.cli.main import main
from quadratic_voting.experiment.test_runner import ScriptedGenerator
from quadratic_voting.experiment.types import GenerationResult, SamplingProfile

class BlockingGenerator(ScriptedGenerator):
    def __init__(self, block_at: int) -> None:
        super().__init__()
        self.block_at = block_at
        self.calls = 0

    def generate(
        self,
        messages: Sequence[ChatMessage],
        profile: SamplingProfile,
        seed: int,
    ) -> GenerationResult:
        self.calls += 1
        if self.calls == self.block_at:
            print(f"READY call={self.calls}", flush=True)
            time.sleep(3600)
        return super().generate(messages, profile, seed)

db, run_id, block_at = sys.argv[1], sys.argv[2], int(sys.argv[3])
raise SystemExit(
    main(
        ["--db", db, "run", "--run-id", run_id],
        generator_factory=lambda _profile: BlockingGenerator(block_at),
    )
)
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed_db = root / "seed.sqlite3"
            seed_store, creation = make_runs(seed_db)
            run_id = creation.run_ids[
                (ElicitationArm.ACTION_ONLY, VotingRegime.OPPOSITION)
            ]
            seed_store.close()

            control_db = root / "control.sqlite3"
            shutil.copy2(seed_db, control_db)
            control = open_sqlite_store(control_db)
            run_experiment(
                run_id,
                store=control,
                generator=ScriptedGenerator(),
                clock=FixedClock(),
            )
            expected = tuple(
                tuple(
                    (message.role.value, message.content)
                    for message in render_transcript(
                        control.voter_round_view(run_id, voter_id)
                    )
                )
                for _index, voter_id in control.voters(run_id)
            )
            control.close()

            for block_at in (1, 2):
                crash_db = root / f"sigkill-{block_at}.sqlite3"
                shutil.copy2(seed_db, crash_db)
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-c",
                        child_code,
                        str(crash_db),
                        str(run_id),
                        str(block_at),
                    ],
                    cwd=Path(__file__).resolve().parents[2],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                try:
                    assert child.stdout is not None
                    ready, _, _ = select.select([child.stdout], [], [], 20)
                    if not ready:
                        assert child.stderr is not None
                        diagnostics = child.stderr.read()
                        self.fail(
                            f"child runner did not reach generation point {block_at}: "
                            f"{diagnostics}"
                        )
                    marker = child.stdout.readline().strip()
                    self.assertEqual(marker, f"READY call={block_at}")
                    os.kill(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
                    self.assertEqual(child.returncode, -signal.SIGKILL)
                finally:
                    if child.poll() is None:
                        child.kill()
                        child.wait(timeout=10)
                    if child.stdout is not None:
                        child.stdout.close()
                    if child.stderr is not None:
                        child.stderr.close()

                # Kernel lock release is immediate: no PID probing, cleanup, or lease wait.
                with acquire_writer_lock(crash_db):
                    pass
                resumed = open_sqlite_store(crash_db)
                started_before = resumed.connection.execute(
                    "SELECT COUNT(*) FROM model_call WHERE status='started'"
                ).fetchone()[0]
                self.assertEqual(started_before, 1)
                if block_at == 2:
                    self.assertGreaterEqual(
                        resumed.connection.execute(
                            "SELECT COUNT(*) FROM model_call WHERE status='committed'"
                        ).fetchone()[0],
                        1,
                    )
                self.assertIs(
                    run_experiment(
                        run_id,
                        store=resumed,
                        generator=ScriptedGenerator(),
                        clock=FixedClock(),
                    ),
                    RunStatus.COMPLETE,
                )
                self.assertEqual(
                    resumed.connection.execute(
                        "SELECT COUNT(*) FROM model_call WHERE status='started'"
                    ).fetchone()[0],
                    0,
                )
                self.assertGreaterEqual(
                    resumed.connection.execute(
                        "SELECT COUNT(*) FROM model_call WHERE status='interrupted'"
                    ).fetchone()[0],
                    1,
                )
                self.assertEqual(
                    resumed.connection.execute(
                        "SELECT turn_id,attempt_index,COUNT(*) FROM model_call "
                        "WHERE status='committed' GROUP BY turn_id,attempt_index "
                        "HAVING COUNT(*) > 1"
                    ).fetchall(),
                    [],
                )
                actual = tuple(
                    tuple(
                        (message.role.value, message.content)
                        for message in render_transcript(
                            resumed.voter_round_view(run_id, voter_id)
                        )
                    )
                    for _index, voter_id in resumed.voters(run_id)
                )
                self.assertEqual(actual, expected)
                resumed.close()


if __name__ == "__main__":
    unittest.main()
