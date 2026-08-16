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
    REGIME_ORDER,
    build_plot_figures,
    build_plot_manifest,
    render_plots,
)
from quadratic_voting.experiment.store import open_sqlite_store
from quadratic_voting.experiment.test_export import AnalysisFixture


EXPECTED = {
    "preference_action_agreement.png",
    "candidate_survival.png",
    "run_quality.png",
    "round_trajectories.png",
    "survival_by_severity.png",
    "net_votes_by_severity.png",
    "ranking_over_rounds.png",
    "vote_share_by_severity.png",
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
            survival = plots["candidate_survival"]
            survival_axis = figures[1][1].axes[0]
            self.assertEqual(survival_axis.get_title(), survival["title"])
            self.assertEqual(survival_axis.get_xlabel(), survival["x_label"])
            self.assertEqual(survival_axis.get_ylabel(), survival["y_label"])
            self.assertEqual(survival_axis.get_ylim(), tuple(survival["y_limits"]))
            self.assertEqual(
                [
                    list(cast(Sequence[float], line.get_xdata()))
                    for line in survival_axis.lines
                ],
                [series["x"] for series in survival["series"]],
            )
            self.assertEqual(
                [
                    list(cast(Sequence[float], line.get_ydata()))
                    for line in survival_axis.lines
                ],
                [series["y"] for series in survival["series"]],
            )
            self.assertEqual(
                [to_hex(line.get_color()) for line in survival_axis.lines],
                [series["color"].lower() for series in survival["series"]],
            )
            self.assertEqual(
                [line.get_linestyle() for line in survival_axis.lines],
                [series["linestyle"] for series in survival["series"]],
            )
            quality = plots["run_quality"]
            quality_left, quality_right = figures[2][1].axes
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
            trajectory = plots["round_trajectories"]
            left, right = figures[3][1].axes
            self.assertEqual(left.get_title(), trajectory["titles"][0])
            self.assertEqual(right.get_title(), trajectory["titles"][1])
            self.assertEqual(
                [list(cast(Sequence[float], line.get_xdata())) for line in left.lines],
                [series["x"] for series in trajectory["series"]],
            )
            self.assertEqual(
                [list(cast(Sequence[float], line.get_ydata())) for line in left.lines],
                [series["active_pool_size"] for series in trajectory["series"]],
            )
            self.assertEqual(
                [line.get_linestyle() for line in left.lines],
                [
                    "-" if series["regime"] == "support" else "--"
                    for series in trajectory["series"]
                ],
            )
            self.assertEqual(
                [to_hex(line.get_color()) for line in left.lines],
                [series["color"].lower() for series in trajectory["series"]],
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
