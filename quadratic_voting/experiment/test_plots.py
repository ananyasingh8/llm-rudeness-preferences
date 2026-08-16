"""Plot smoke tests against populated and schema-only Parquet exports."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from matplotlib.patches import Rectangle
from matplotlib.colors import to_hex

from quadratic_voting.experiment.export import export_parquet
from quadratic_voting.experiment.plots import (
    ARM_ORDER,
    PALETTE,
    PLOT_MANIFEST_VERSION,
    REGIME_LINESTYLE,
    REGIME_ORDER,
    build_plot_figures,
    build_plot_manifest,
    render_plots,
)
from quadratic_voting.experiment.plots import (
    _severity_palette_manifest,
)
from quadratic_voting.experiment.store import open_sqlite_store
from quadratic_voting.experiment.test_export import AnalysisFixture


SEVERITY_PALETTE_MANIFEST = _severity_palette_manifest()


EXPECTED = {
    "preference_action_agreement.png",
    "run_quality.png",
    "vote_share_by_severity.png",
    "net_votes_by_severity.png",
    "candidate_survival.png",
    "round_trajectories.png",
    "pooled_by_severity.parquet",
    "timeline.html",
}


class FixturePlotTests(AnalysisFixture):
    def test_fixture_exports_render_expected_pngs(self) -> None:
        export_dir = self.root / "exports"
        export_parquet(self.export_store, export_dir)
        paths = render_plots(export_dir, self.root / "plots")
        self.assertEqual({path.name for path in paths}, EXPECTED)
        self.assertTrue(all(path.stat().st_size > 0 for path in paths))
        timeline = (self.root / "plots" / "timeline.html").read_text(encoding="utf-8")
        self.assertIn("Quadratic-voting allocation flows", timeline)
        self.assertIn("Conversation actually shown in this pilot", timeline)
        manifest_path = self.root / "plots" / "plot-manifest.json"
        self.assertTrue(manifest_path.exists())
        self.assertEqual(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            build_plot_manifest(export_dir),
        )

    def test_semantic_manifest_and_artists_pin_order_labels_limits_and_style(
        self,
    ) -> None:
        export_dir = self.root / "exports"
        export_parquet(self.export_store, export_dir)
        manifest = build_plot_manifest(export_dir)
        self.assertEqual(manifest["version"], PLOT_MANIFEST_VERSION)
        self.assertEqual(manifest["palette"], PALETTE)
        # Shared style is exposed in the manifest with JSON-safe string keys.
        self.assertEqual(manifest["severity_palette"], SEVERITY_PALETTE_MANIFEST)
        self.assertEqual(manifest["regime_linestyle"], REGIME_LINESTYLE)
        plots = manifest["plots"]
        assert isinstance(plots, dict)
        agreement = plots["preference_action_agreement"]
        categories = agreement["categories"]
        self.assertEqual(
            categories,
            sorted(
                categories,
                key=lambda label: (
                    label.split("\n")[0],
                    int(label.split("\n")[1][1:]),
                    ARM_ORDER.index(label.split("\n")[2]),
                    REGIME_ORDER.index(label.split("\n")[3]),
                ),
            ),
        )
        figures = build_plot_figures(manifest)
        try:
            # figures[0] — agreement (unchanged bar chart)
            agreement_axis = figures[0][1].axes[0]
            self.assertEqual(agreement_axis.get_ylim(), (-1.0, 1.0))
            self.assertEqual(
                agreement_axis.get_ylabel(),
                "Mean within-voter-round Spearman rho",
            )
            self.assertEqual(
                [
                    cast(Rectangle, patch).get_height()
                    for patch in agreement_axis.patches
                ],
                agreement["values"],
            )
            self.assertEqual(
                [
                    to_hex(cast(Rectangle, patch).get_facecolor())
                    for patch in agreement_axis.patches
                ],
                [color.lower() for color in agreement["colors"]],
            )
            # figures[1] — run_quality (reliability stack + error-code bars)
            quality = plots["run_quality"]
            quality_left, quality_right = figures[1][1].axes
            self.assertEqual(
                [cast(Rectangle, patch).get_height() for patch in quality_left.patches],
                [
                    value
                    for values in quality["reliability_metrics"].values()
                    for value in values
                ],
            )
            self.assertEqual(
                [
                    round(cast(Rectangle, patch).get_width())
                    for patch in quality_right.patches
                ],
                quality["error_codes"]["counts"],
            )
            # figures[2] — vote share: one line per (severity, regime); color =
            # severity, linestyle = regime; y in [0, 1].
            vote_share = plots["vote_share_by_severity"]
            vote_axis = figures[2][1].axes[0]
            self.assertEqual(vote_axis.get_ylim(), (0.0, 1.0))
            self.assertEqual(vote_axis.get_ylabel(), vote_share["y_label"])
            vote_series = vote_share["series"]
            data_lines = [
                line
                for line in vote_axis.lines
                if len(cast("Sequence[float]", line.get_xdata())) > 0
            ]
            self.assertEqual(len(data_lines), len(vote_series))
            self.assertEqual(
                [to_hex(line.get_color()) for line in data_lines],
                [series["color"].lower() for series in vote_series],
            )
            self.assertEqual(
                [line.get_linestyle() for line in data_lines],
                [series["linestyle"] for series in vote_series],
            )
            # figures[4] — candidate survival: left pooled + right per-run panels.
            survival = plots["candidate_survival"]
            survival_left, survival_right = figures[4][1].axes
            self.assertEqual(survival_left.get_title(), survival["pooled"]["title"])
            self.assertEqual(survival_right.get_title(), survival["per_run"]["title"])
            # figures[5] — round trajectories: left pooled + right per-run panels.
            trajectory = plots["round_trajectories"]
            traj_left, traj_right = figures[5][1].axes
            self.assertEqual(traj_left.get_title(), trajectory["pooled"]["title"])
            self.assertEqual(traj_right.get_title(), trajectory["per_run"]["title"])
            per_run_series = trajectory["per_run"]["series"]
            traj_data_lines = [
                line
                for line in traj_right.lines
                if len(cast("Sequence[float]", line.get_xdata())) > 0
            ]
            self.assertEqual(len(traj_data_lines), len(per_run_series))
            self.assertEqual(
                [line.get_linestyle() for line in traj_data_lines],
                [series["linestyle"] for series in per_run_series],
            )
        finally:
            import matplotlib.pyplot as plt

            for _, figure in figures:
                plt.close(figure)


class EmptyPlotTests(unittest.TestCase):
    def test_plot_failure_never_publishes_partial_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with open_sqlite_store(root / "empty.sqlite3") as store:
                export_parquet(store, root / "exports")
            with mock.patch(
                "matplotlib.figure.Figure.savefig",
                side_effect=OSError("injected plot write failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected plot"):
                    render_plots(root / "exports", root / "failed-plots")
            self.assertFalse((root / "failed-plots").exists())
            self.assertEqual(list(root.glob(".failed-plots.staging-*")), [])

    def test_empty_store_exports_render_annotated_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with open_sqlite_store(root / "empty.sqlite3") as store:
                export_parquet(store, root / "exports")
            paths = render_plots(root / "exports", root / "plots")
            self.assertEqual({path.name for path in paths}, EXPECTED)
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))


if __name__ == "__main__":
    unittest.main()
