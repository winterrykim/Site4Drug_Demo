import sys
import types

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")

    class _RequestException(Exception):
        pass

    def _unavailable(*args, **kwargs):
        raise _RequestException("requests unavailable in test harness")

    requests_stub.RequestException = _RequestException
    requests_stub.get = _unavailable
    sys.modules["requests"] = requests_stub

import unittest

from site4drug_inference.common.site_output_schema import validate_site_output, with_schema_defaults
from site4drug_inference.demo import predict_site


class SchemaAndReportTests(unittest.TestCase):
    def test_schema_defaults_do_not_restore_removed_runtime_fields(self):
        payload = with_schema_defaults(
            {
                "recommended_modality": "epitope",
                "modality_confidence": 0.5,
                "ranked_candidates": [{"rank": 1, "candidate_id": "L_C0001", "start": 1, "end": 4, "peptide": "AAAA"}],
                "candidate_evidence": [{"candidate_id": "L_C0001", "evidence": []}],
                "risk_flags": [],
                "agent_traces": {},
                "feature_provenance": {
                    "ptm_source": "musitedeep",
                    "ptm_rule_version": "rulepack_v1",
                    "motif_source": "remote",
                    "motif_library_version": "scanprosite_biopython_v1",
                    "motif_remote_status": "ok",
                },
                "token_strategy_used": "full_sequence",
                "audit_log": {"warnings": [], "events": []},
                "ptm_summary": {},
                "motif_summary": {},
                "iedb_validation": {},
                "orchestrator_trace": [],
            }
        )
        row = payload["ranked_candidates"][0]
        self.assertNotIn("topology_label", row)
        self.assertNotIn("accessibility_label", row)
        self.assertEqual(validate_site_output(payload), [])

    def test_compact_reports_preserve_current_sections(self):
        run = {
            "run_id": "demo_run",
            "run_status": "ok",
            "input": {"uniprot": "P29996", "mode_request": "auto"},
            "recommended_modality": "pocket",
            "modality_confidence": 0.82,
            "generation": {
                "candidate_source": "llm_propose",
                "ptm_backend_effective": "musitedeep",
                "motif_source_effective": "remote",
            },
            "ranked_candidates": [
                {
                    "rank": 1,
                    "candidate_id": "L_C0001",
                    "mode": "pocket",
                    "peptide": "AAAA",
                    "start": 10,
                    "end": 13,
                    "confidence": "High",
                    "confidence_score": 0.82,
                    "confidence_source": "llm_self_consistency_vote",
                    "flags": ["motif-overlap"],
                    "reason": "Consensus candidate.",
                }
            ],
            "agent_traces": {},
            "ptm_summary": {"total_sites": 1},
            "motif_summary": {"total_hits": 1},
            "raw_api_calls": {
                "musitedeep": {"status": "ok", "request_count": 1, "artifact_path": "musitedeep_raw.json", "preview": "..."},
                "scanprosite": {"status": "ok", "n_hits": 1, "artifact_path": "scanprosite_raw.xml", "preview": "..."},
            },
            "plot_artifacts": {"plot_png_name": "hydropathy_ptm_plot.png"},
        }
        md = predict_site._render_markdown_report_compact(run)
        html = predict_site._render_html_report_compact(run)
        self.assertIn("## Ranked Candidates", md)
        self.assertIn("## PTM + Motif Summary", md)
        self.assertIn("Hydropathy + PTM + Candidate Tracks", md)
        self.assertIn("Prediction Report", html)
        self.assertIn("Ranked Candidates", html)
        self.assertNotIn("Deterministic Bioinformatics Baseline", md)


if __name__ == "__main__":
    unittest.main()
