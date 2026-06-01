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

from site4drug_inference.demo import predict_site


class SelfConsistencyAndFillTests(unittest.TestCase):
    def test_self_consistency_prefers_consensus_cluster(self):
        attempts = [
            {
                "attempt_index": 1,
                "parsed_obj": {"recommended_modality": "pocket", "modality_confidence": 0.8},
                "ranked_candidates": [
                    {"rank": 1, "candidate_id": "A1", "mode": "pocket", "start": 10, "end": 13, "peptide": "AAAA", "confidence_score": 0.8, "heuristic_score": 0.7, "flags": []}
                ],
            },
            {
                "attempt_index": 2,
                "parsed_obj": {"recommended_modality": "pocket", "modality_confidence": 0.7},
                "ranked_candidates": [
                    {"rank": 1, "candidate_id": "A2", "mode": "pocket", "start": 11, "end": 14, "peptide": "AAAB", "confidence_score": 0.7, "heuristic_score": 0.6, "flags": []}
                ],
            },
            {
                "attempt_index": 3,
                "parsed_obj": {"recommended_modality": "epitope", "modality_confidence": 0.6},
                "ranked_candidates": [
                    {"rank": 1, "candidate_id": "B1", "mode": "epitope", "start": 40, "end": 51, "peptide": "BBBBBBBBBBBB", "confidence_score": 0.6, "heuristic_score": 0.4, "flags": []}
                ],
            },
        ]
        consensus = predict_site._build_self_consistency_consensus(attempts, requested_k=3, top_k=2)
        self.assertEqual(consensus["recommended_modality"], "pocket")
        self.assertEqual(consensus["ranked_candidates"][0]["candidate_id"], "L_C0001")
        self.assertIn("self_consistency_votes_2_of_3", consensus["ranked_candidates"][0]["flags"])

    def test_llm_validation_does_not_backfill_missing_rows(self):
        parsed_obj = {
            "recommended_modality": "epitope",
            "ranked_candidates": [
                {"rank": 1, "start": 1, "end": 12, "peptide": "ACDEFGHIKLMN", "mode": "epitope", "confidence_score": 0.9, "reason": "Valid."}
            ],
        }
        seq_summary = {
            "tm_regions": [],
            "ptm_sites": [],
            "motif_hits": [],
            "cysteine_positions": [],
        }
        deterministic_candidates = [
            {"candidate_id": "D1", "mode": "epitope", "start": 20, "end": 31, "peptide": "QRSTVWYACDEF", "risk_flags": []}
        ]
        parsed, stats = predict_site._validate_and_enrich_llm_proposals(
            parsed_obj=parsed_obj,
            sequence="ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
            requested_mode="epitope",
            top_k=2,
            seq_summary=seq_summary,
            deterministic_candidates=deterministic_candidates,
            ptm_policy="tiered",
        )
        self.assertEqual(len(parsed["ranked_candidates"]), 1)
        self.assertEqual(stats["llm_proposal_fill_count"], 0)
        self.assertEqual(stats["llm_proposal_valid"], 1)

    def test_llm_enrichment_counts_typed_ptm_mask_overlap(self):
        seq_summary = {
            "tm_regions": [],
            "ptm_sites": [
                {
                    "ptm_type": "N-linked_glycosylation",
                    "position": 10,
                    "mask_start": 7,
                    "mask_end": 13,
                    "rule_confidence": "high",
                }
            ],
            "motif_hits": [],
            "cysteine_positions": [],
        }
        enriched = predict_site._enrich_span_candidate(
            sequence="ACDEFGHIKLMNPQRSTVWY",
            start=4,
            end=8,
            mode="epitope",
            seq_summary=seq_summary,
            ptm_policy="tiered",
        )
        self.assertTrue(enriched["overlaps_ptm_mask"])
        self.assertEqual(enriched["ptm_overlap_by_type"], {"N-linked_glycosylation": 1})
        self.assertIn("PTM-overlap", enriched["risk_flags"])
        self.assertIn("glyco-mask-overlap", enriched["risk_flags"])


if __name__ == "__main__":
    unittest.main()
