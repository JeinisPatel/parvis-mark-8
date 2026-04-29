"""
model.py rebuild — Chapter 5 (April 11, 2026 canonical) taxonomy.

Migration strategy (per JP confirmation):
  - Preserve current CPT calibrations where there's an obvious content match
  - Conservative defaults for new CH5 nodes (N12, N13, N16, N18)
  - These values are illustrative anchors for the constructive proof, not 
    empirical calibrations — per JP "remember these are just estimates based 
    on the purpose of the thesis as a constructive proof of concept"

Node remapping table (CURRENT → CH5 CANONICAL):
  N1  Burden of proof                  → N1  (same)
  N2  Violent history                  → N2  (Validated Risk Elevators)
  N3  PCL-R psychopathy                → N3  (folds into Sexual Offence Risk Profile)
  N4  Static-99R                       → N3  (also folds into Sexual Offence Risk Profile)
  N5  Invalid risk tools               → N5  (same)
  N6  IAC                              → N6  (same)
  N7  Bail-WCGP cascade                → N7  (same)
  N8  King impeachment                 → N7i (becomes sub-node, kept as separate node N8 for now)
  N9  FASD                             → N8  (CH5 §5.1.8)
  N10 Intergenerational trauma         → N9  (consolidated into IGT & Cultural Treatment)
  N11 No cultural treatment            → N9  (folds into N9 in CH5)
  N12 Gladue misapplication            → N10 (Judicial Misapplication of SCE)
  N13 Gaming risk                      → N11 (Gaming Risk Detector)
  N14 Over-policing                    → N17 (Over-Policing and Epistemic Contamination)
  N15 Temporal distortion              → N14 (Temporal Distortion)
  N16 Tariff disparities               → N15 (Interjurisdictional Tariff Distortion)
  N17 Collider bias                    → N19 (Collider Bias)
  N18 Dynamic risk                     → N4  (Dynamic Risk Factor Cluster)
  N19 No rehabilitation                → N9  (folds into N9 in CH5)
  N20 DO designation                   → N20 (same)

NEW nodes in CH5 not in current architecture:
  N12 Judicial Reasoning Reliability (Judging the Judge)
  N13 Structural Systemic Bias (TraceRoute)
  N16 Doctrinal Tension (s.718.04 vs s.718.2(e))
  N18 Gladue/Ewert/Morris/Ellis Profile audit

Note on consolidation: current N10 (IGT), N11 (treatment), N19 (rehab) all 
map to CH5 N9. Since the BN structure can't have three nodes mapping to one,
the rebuild creates ONE N9 in the new structure that represents the 
consolidated concept. The CPT for N9 is taken from current N10's calibration
since IGT is conceptually primary; treatment-availability becomes part of 
the N9 specification rather than a separate node.

The current N3 (PCL-R) and N4 (Static-99R) similarly both map to CH5 N3
(Sexual Offence Risk Profile). The rebuild treats CH5 N3 as the consolidated
sexual-offence-profile node and uses current N4's CPT (since Static-99R is 
the more directly sexual-offence-specific instrument). PCL-R is then 
implicit within N3.
"""

from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# Edges in the canonical Chapter 5 taxonomy
# ═════════════════════════════════════════════════════════════════════════════
#
# Parent → Child relationships derived from CH5 §5.1.X specifications.
# Topology must be a DAG (no cycles).
#
# Structural notes:
#   N1 (Burden of proof) is upstream conditioner of evidentiary nodes
#   N7 → N2 cascade preserved (bail-WCGP affects record reliability)
#   N20 (DO designation) is downstream of all distortion + risk nodes
#       but computed POST-VE (not in pgmpy network)
# Edges with int IDs and string IDs both supported. String IDs are used for
# the §5.1.17 sub-nodes (17a/b/c/d) and §5.1.14 sub-nodes (14a/b/c/d).
EDGES_VE = [(str(f), str(t)) for f, t in [
    # Layer I — Substantive Risk
    (1, 2), (1, 3), (1, 4),
    # Layer II — Distortion conditioning
    (1, 6), (1, 8),                 # N1 conditions IAC and FASD evidentiary thresholds
    (7, 2),                         # bail-WCGP affects violent-history reliability (preserved)
    (2, 5), (3, 5), (4, 5),        # validated risk inputs feed risk-tool node
    # NOTE: (2, 14) DROPPED per Q2 — N2 not doctrinally a parent of N14.
    # N14's parents are §5.1.14 §5 production conditions (era, MM, SCE, judicial competence).
    (6, 7),                         # IAC → bail cascade
    (6, 10),                        # IAC contributes to judicial misapplication
    (8, 9),                         # FASD → IGT/cultural-treatment
    (9, 10),                        # IGT → judicial misapplication
    (5, 10),                        # invalid risk tools → judicial misapplication
    (10, 12),                       # judicial misapplication → judging the judge
    # NOTE: (10, 18) and (9, 18) DROPPED per N18 Q5 (α) — neither is a §5.1.18
    # §5 parent at the node level. N10 routes into N18 via N18d kinship edge
    # (added below); N9 (IGT) is not a §5.1.18 §5 parent at all and was a
    # holdover from the earlier Noisy-OR cpd18 specification.
    (3, 11),                        # sexual offence risk profile → gaming detector
    # ── §5.1.17 N17 four-parent topology (C3: TraceRoute routes through N17a) ──
    (13, '17a'),                    # TraceRoute → Jurisdictional Policing Disparity
    ('17a', 17), ('17b', 17), ('17c', 17), ('17d', 17),  # 4 parents → N17
    (17, 19),                       # over-policing → collider bias
    (14, 19),                       # temporal distortion → collider bias  
    (10, 16),                       # judicial misapplication → doctrinal tension
    # NOTE: (10, 15) DROPPED per Q4 — N10 not a doctrinal parent of N15 per
    # §5.1.15 §5. The five canonical parents are jurisdiction, offence type,
    # sentence length, year, and jurisprudential weighting — N10 doesn't
    # appear among them. Old edge was holdover from earlier taxonomy.
    (14, 18),                       # temporal distortion → SCE Profile audit
    (12, 18),                       # judging-the-judge → SCE Profile audit
    # ── §5.1.18 N18 four-sub-node + N12/N14 topology (Q5 (α)) ──
    # N13 → N18a routes structural-bias signal through jurisdiction sensitivity
    # sub-node (mirrors N13→N14a, N13→N15a, N13→N17a). N10 → N18d carries
    # judicial-misapplication signal into Doctrinal Tagging compliance
    # (mirrors N10→N14d kinship). Existing (14,18) and (12,18) edges retained
    # at node level — N14 (temporal) and N12 (judging-the-judge) are §5.1.18
    # §5 parents 5 and 4 respectively, treated as structural amplifiers.
    (13, '18a'),                    # TraceRoute → Jurisdiction sensitivity (structural prior)
    (10, '18d'),                    # Misapplication → Doctrinal Tagging compliance (kinship)
    ('18a', 18), ('18b', 18), ('18c', 18), ('18d', 18),  # 4 sub-nodes → N18
    # ── §5.1.14 N14 four-parent topology (Q3: N13→14a, N10→14d) ──
    (13, '14a'),                    # TraceRoute → Era Severity (structural prior)
    (10, '14d'),                    # Misapplication → Judicial Competence Absent (kinship)
    ('14a', 14), ('14b', 14), ('14c', 14), ('14d', 14),  # 4 parents → N14
    # ── §5.1.15 N15 four-sub-node + N14 topology (Q4 (α)) ──
    # N13 → N15a routes structural-bias signal through tariff-jurisdiction sub-node
    # (mirrors N13→N14a, N13→N17a). N14 → N15 directly at node level honors
    # §5.1.15's explicit downstream-of-N14 positioning (Position section).
    (13, '15a'),                    # TraceRoute → Tariff jurisdiction (structural prior)
    (14, 15),                       # Temporal distortion → Tariff distortion (§5.1.15 Position)
    ('15a', 15), ('15b', 15), ('15c', 15), ('15d', 15),  # 4 sub-nodes → N15
]]


# ═════════════════════════════════════════════════════════════════════════════
# Node metadata — CH5 (April 11, 2026) canonical taxonomy
# ═════════════════════════════════════════════════════════════════════════════
NODE_META = {
    # ── Substantive Risk Layer (CH5 i) ────────────────────────────────────────
    1:  {"name": "Criminal law burden of proof",                  "short": "Burden of proof",       "type": "constraint", "ev": False},
    2:  {"name": "Validated risk elevators",                      "short": "Risk elevators",        "type": "risk",       "ev": True},
    3:  {"name": "Sexual offence risk profile",                   "short": "Sexual offence",        "type": "risk",       "ev": True},
    4:  {"name": "Dynamic risk factor cluster",                   "short": "Dynamic risk",          "type": "risk",       "ev": True},
    # ── Systemic Distortion and Doctrinal Fidelity Layer (CH5 ii) ────────────
    5:  {"name": "Current risk assessment tools",                 "short": "Risk tools",            "type": "distortion", "ev": True},
    6:  {"name": "Ineffective assistance of counsel",             "short": "IAC",                   "type": "distortion", "ev": True},
    7:  {"name": "Bail denial → wrongful conviction guilty plea", "short": "Bail-WCGP cascade",     "type": "distortion", "ev": True},
    8:  {"name": "FASD as dual-factor in risk modeling",          "short": "FASD",                  "type": "dual",       "ev": True},
    9:  {"name": "Intergenerational trauma & cultural treatment", "short": "IGT / treatment",       "type": "mitigation", "ev": True},
    10: {"name": "Judicial misapplication of SCE",                "short": "SCE misapplication",    "type": "distortion", "ev": True},
    11: {"name": "Gaming risk detector",                          "short": "Gaming risk",           "type": "special",    "ev": True},
    12: {"name": "Judicial reasoning reliability",                "short": "Judging the judge",     "type": "distortion", "ev": False},
    13: {"name": "Structural systemic bias (TraceRoute)",         "short": "TraceRoute",            "type": "distortion", "ev": False},
    14: {"name": "Temporal distortion in prior records",          "short": "Temporal distortion",   "type": "distortion", "ev": True},
    15: {"name": "Interjurisdictional tariff distortion",         "short": "Tariff distortion",     "type": "distortion", "ev": False},
    16: {"name": "Doctrinal tension (s.718.04 / s.718.2(e))",     "short": "Doctrinal tension",     "type": "distortion", "ev": False},
    17: {"name": "Over-policing & epistemic contamination",       "short": "Over-policing",         "type": "distortion", "ev": True},
    18: {"name": "Gladue / Ewert / Morris / Ellis profile",       "short": "SCE profile audit",     "type": "distortion", "ev": False},
    19: {"name": "Collider bias",                                 "short": "Collider bias",         "type": "distortion", "ev": False},
    # ── §5.1.17 sub-nodes — four parents of N17 (added per JP confirmation C1) ──
    # These are sub-nodes of N17 (similar to N10a-N10d for Gladue misapplication factors).
    # IDs stored as strings to disambiguate from int node IDs.
    "17a": {"name": "Jurisdictional policing disparity",          "short": "Disparity",       "type": "distortion", "ev": True},
    "17b": {"name": "Enforcement-disparity engagement",           "short": "Engagement",  "type": "distortion", "ev": True},
    "17c": {"name": "Non-violent charge density",                 "short": "Non-violent",   "type": "distortion", "ev": True},
    "17d": {"name": "Surveillance-triggered entries",             "short": "Surveillance",          "type": "distortion", "ev": True},
    # ── §5.1.14 sub-nodes — four parents of N14 (Q1: mirror N17 pattern) ──
    # N14a: Sentencing era severity (state 1 = severe era)
    # N14b: Historical mandatory minimum (state 1 = MM-era)
    # N14c: SCE absent at sentencing (state 1 = absent — adverse direction)
    # N14d: Judicial competence absent (state 1 = absent — adverse direction)
    "14a": {"name": "Sentencing era severity",                    "short": "Era severity",          "type": "distortion", "ev": True},
    "14b": {"name": "Historical mandatory minimum",               "short": "Mandatory min",         "type": "distortion", "ev": True},
    "14c": {"name": "SCE absent at sentencing",                   "short": "SCE absent",            "type": "distortion", "ev": True},
    "14d": {"name": "Judicial competence absent",                 "short": "Comp absent",           "type": "distortion", "ev": True},
    # ── §5.1.15 sub-nodes — four parents of N15 (Q2: mirror N14 pattern) ──
    # N15a: Tariff jurisdiction (state 1 = High-tariff jurisdiction)
    # N15b: Tariff offence (state 1 = tariff-sensitive offence category)
    # N15c: Tariff length (state 1 = sentence exceeds offence-conditional threshold)
    # N15d: Doctrine absent (state 1 = no jurisprudence applied — adverse direction)
    "15a": {"name": "Tariff jurisdiction disparity",              "short": "Tariff jurisdiction",   "type": "distortion", "ev": True},
    "15b": {"name": "Tariff-sensitive offence type",              "short": "Tariff offence",        "type": "distortion", "ev": True},
    "15c": {"name": "Tariff-sensitive sentence length",           "short": "Tariff length",         "type": "distortion", "ev": True},
    "15d": {"name": "Jurisprudential compliance absent",          "short": "Doctrine absent",       "type": "distortion", "ev": True},
    # ── §5.1.18 sub-nodes — four parents of N18 (Q2 (α): mirror N14/N15/N17) ──
    # N18a: Jurisdiction sensitivity (state 1 = no Morris/Ellis precedent)
    # N18b: SCE Presence in Reasons (state 1 = absent in reasons aggregate)
    # N18c: SCE Substance (state 1 = nominal-only or absent — Morris Audit)
    # N18d: Doctrinal Tagging compliance (state 1 = incomplete or error)
    "18a": {"name": "Jurisdiction SCE-integration sensitivity",   "short": "Jurisdiction sensitivity", "type": "distortion", "ev": True},
    "18b": {"name": "SCE presence in reasons",                    "short": "SCE presence",          "type": "distortion", "ev": True},
    "18c": {"name": "SCE substance",                              "short": "SCE substance",         "type": "distortion", "ev": True},
    "18d": {"name": "Doctrinal tagging compliance",               "short": "Doctrinal tagging",     "type": "distortion", "ev": True},
    # ── Structural Output (CH5 iii) ──────────────────────────────────────────
    20: {"name": "Dangerous offender designation",                "short": "DO designation",        "type": "output",     "ev": False},
}


# Backward-compat alias for any code expecting EDGES (not EDGES_VE)
EDGES = EDGES_VE


def _cpt(var, parents, table):
    """Helper: build TabularCPD. table = [P(Low|...), P(High|...)] columns."""
    n_parents = len(parents)
    ev_card = [2] * n_parents
    return TabularCPD(
        variable=var,
        variable_card=2,
        values=table,
        evidence=parents if parents else None,
        evidence_card=ev_card if ev_card else None,
    )


def _noisy_or(var, parents, leak, inhibitors):
    """
    Noisy-OR CPT for a node with many parents.
    P(High) = 1 - leak * prod(inhibitors[i] if parent_i=High else 1)
    """
    n = len(parents)
    n_configs = 2 ** n
    p_high = np.zeros(n_configs)

    for config in range(n_configs):
        bits = [(config >> i) & 1 for i in range(n)]
        q = leak
        for i, active in enumerate(bits):
            if active:
                q *= inhibitors[i]
        p_high[config] = 1.0 - q

    values = [1 - p_high, p_high]
    return TabularCPD(
        variable=var,
        variable_card=2,
        values=values,
        evidence=parents,
        evidence_card=[2] * n,
    )


def build_model():
    """Build and return the PARVIS BayesianNetwork with all CPTs.
    Node 20 is excluded from the pgmpy network — it is computed post-VE.
    
    CPT calibrations are illustrative anchors per the constructive-proof
    nature of the thesis. Where a current-architecture CPT has an obvious
    canonical match, calibration is preserved. New nodes (N12, N13, N16, N18)
    receive conservative defaults reflecting the doctrinal posture in CH5.
    """
    model = BayesianNetwork([(str(f), str(t)) for f, t in EDGES_VE])

    # ── N1: Criminal Law Burden of Proof (root) ──────────────────────────────
    # CH5 §5.1.1 — evidentiary admissibility constraint at BRD level (~83%)
    cpd1 = TabularCPD(variable='1', variable_card=2, values=[[0.17], [0.83]])

    # ── N2: Validated Risk Elevators (parents: N1, N7) ───────────────────────
    # CPT preserved from current N2 (was Violent history)
    # Parent topology (N1, N7) preserved — bail-WCGP cascade discounts record
    cpd2 = _cpt('2', ['1', '7'], [
        [0.50, 0.55, 0.35, 0.55],
        [0.50, 0.45, 0.65, 0.45],
    ])

    # ── N3: Sexual Offence Risk Profile (parent: N1) ─────────────────────────
    # Consolidates current N3 (PCL-R) + N4 (Static-99R) per CH5 §5.1.3
    # CPT preserved from current N4 (Static-99R) — more directly applicable
    cpd3 = _cpt('3', ['1'], [
        [0.80, 0.65],
        [0.20, 0.35],
    ])

    # ── N4: Dynamic Risk Factor Cluster (parent: N1) ─────────────────────────
    # Maps from current N18 (Dynamic risk) per CH5 §5.1.4
    # Preserved CPT structure — single parent (N1) for CH5 Layer I structure
    # Note: current N18 had three parents (N11, N13, N15) but in CH5 N4 is
    # an upstream substantive risk node, so the parent set is reduced to N1.
    cpd4 = _cpt('4', ['1'], [
        [0.55, 0.30],
        [0.45, 0.70],
    ])

    # ── N5: Current Risk Assessment Tools (parents: N2, N3, N4) ──────────────
    # CPT preserved from current N5 (parents same)
    cpd5 = _cpt('5', ['2', '3', '4'], [
        [0.35, 0.22, 0.25, 0.15, 0.28, 0.18, 0.20, 0.10],
        [0.65, 0.78, 0.75, 0.85, 0.72, 0.82, 0.80, 0.90],
    ])

    # ── N6: IAC (parent: N1) ─────────────────────────────────────────────────
    # CPT preserved from current N6
    cpd6 = _cpt('6', ['1'], [
        [0.60, 0.45],
        [0.40, 0.55],
    ])

    # ── N7: Bail Denial → WCGP Cascade (parent: N6) ──────────────────────────
    # CPT preserved from current N7
    cpd7 = _cpt('7', ['6'], [
        [0.55, 0.30],
        [0.45, 0.70],
    ])

    # ── N8: FASD as Dual-Factor (parent: N1) ─────────────────────────────────
    # Maps from current N9 (FASD), CPT preserved
    cpd8 = _cpt('8', ['1'], [
        [0.70, 0.60],
        [0.30, 0.40],
    ])

    # ── N9: IGT and Cultural Treatment (parent: N8) ──────────────────────────
    # Consolidates current N10 (IGT) + N11 (treatment) + N19 (rehab) per CH5 §5.1.9
    # CPT preserved from current N10 (IGT was conceptually primary)
    cpd9 = _cpt('9', ['8'], [
        [0.35, 0.15],
        [0.65, 0.85],
    ])

    # ── N10: Judicial Misapplication of SCE (parents: N5, N6, N9) ────────────
    # Maps from current N12 (Gladue misapplication) per CH5 §5.1.10
    # Parent set adjusted for canonical taxonomy: invalid tools (N5),
    # ineffective counsel (N6), and IGT/treatment (N9) all condition
    # whether SCE was meaningfully applied. Three-parent CPT.
    # 8 combinations: (5,6,9) = (L,L,L),(H,L,L),(L,H,L),(H,H,L),(L,L,H),(H,L,H),(L,H,H),(H,H,H)
    # Calibration derived from current 4-parent N12 CPT, marginalised over the
    # dropped parents conservatively.
    cpd10 = _cpt('10', ['5', '6', '9'], [
        [0.55, 0.40, 0.42, 0.30, 0.40, 0.25, 0.30, 0.18],
        [0.45, 0.60, 0.58, 0.70, 0.60, 0.75, 0.70, 0.82],
    ])

    # ── N11: Gaming Risk Detector (parent: N3) ───────────────────────────────
    # Maps from current N13 (Gaming risk detector), CPT preserved
    cpd11 = _cpt('11', ['3'], [
        [0.82, 0.65],
        [0.18, 0.35],
    ])

    # ── N12: Judicial Reasoning Reliability (parent: N10) ────────────────────
    # NEW node per CH5 §5.1.12 — "Judging the Judge"
    # Conservative default: when SCE has been misapplied (N10 High), 
    # judicial reasoning reliability is reduced (P(High reasoning reliability) lower)
    # Note: "High" state of N12 is interpreted as HIGH reliability concern,
    # consistent with state="High" = distortion/concern present elsewhere.
    cpd12 = _cpt('12', ['10'], [
        [0.65, 0.40],     # P(Low) — high reliability when SCE properly applied
        [0.35, 0.60],     # P(High) — reliability concern when SCE misapplied
    ])

    # ── N13: Structural Systemic Bias / TraceRoute (no parents) ──────────────
    # NEW node per CH5 §5.1.13 — root systemic-bias node
    # Conservative default reflects the empirical baseline of structural 
    # discrimination in the Canadian justice system per CH3 evidence base.
    cpd13 = TabularCPD(variable='13', variable_card=2, values=[[0.40], [0.60]])

    # ── N14: Temporal Distortion (parent: N2) ────────────────────────────────
    # Maps from current N15 (Temporal distortion), CPT preserved
    # ── N14a: Sentencing era severity (parent: N13 TraceRoute) ──────────────
    # Per §5.1.14 §5 + Q3 routing: TraceRoute (structural systemic bias) conditions
    # the era severity index. When N13 (TraceRoute) is High, structural-bias-era
    # conditions raise baseline expectation that the sentencing era was severe.
    cpd14a = _cpt('14a', ['13'], [
        [0.65, 0.40],     # P(era severity Low) — neutral when N13 Low; corrected when High
        [0.35, 0.60],     # P(era severity High) — rises with structural bias signal
    ])

    # ── N14b: Historical mandatory minimum (no parents — evidence) ──────────
    # Per Q6 (α): pattern-matched from criminal record (offence type + year).
    # Default prior reflects baseline rate of MM-era convictions in records.
    cpd14b = TabularCPD(variable='14b', variable_card=2, values=[[0.65], [0.35]])

    # ── N14c: SCE absent at sentencing (no parents — evidence) ──────────────
    # Per Q6 (α): pattern-matched from conviction years (pre-2012 = SCE typically
    # absent or nominal; post-2012 = SCE formally available per Ipeelee).
    # Default prior reflects baseline rate.
    cpd14c = TabularCPD(variable='14c', variable_card=2, values=[[0.55], [0.45]])

    # ── N14d: Judicial competence absent (parent: N10) ──────────────────────
    # Per Q3: N10 (Judicial Misapplication) conditions cultural competence.
    # When misapplication is High, judicial competence at original sentencing
    # was likely absent or superficial.
    cpd14d = _cpt('14d', ['10'], [
        [0.70, 0.35],     # P(competence absent = Low) when N10 Low / High
        [0.30, 0.65],     # P(competence absent = High) — rises with N10 high
    ])

    # ── N14: Temporal distortion (4 parents) ────────────────────────────────
    # Per CH5 §5.1.14 §7 illustrative CPT (binary collapse, anchors hit exactly):
    #   (0,0,0,0) all benign            → P(distortion) = 0.15  [§7 row 4]
    #   (1,1,1,1) all adverse           → P(distortion) = 0.95  [§7 row 1]
    #   (1,1,0,1) severe era + MM +     → P(distortion) = 0.90  [§7 row 2,
    #     SCE present + comp absent       binary collapse: superficial→absent]
    #
    # Big-endian column ordering per pgmpy:
    #   col_idx = state(14a)*8 + state(14b)*4 + state(14c)*2 + state(14d)*1
    #
    # Structural weighting per §5.1.14 §4: era severity + MM are the dominant
    # adverse signals (production-condition factors); SCE and competence are
    # secondary (correctable-at-time-of-sentencing factors). Single-parent
    # contributions: N14a ~0.30, N14b ~0.25, N14c ~0.15, N14d ~0.15.
    # Anchor points hit §5.1.14 §7 exactly; intermediate combinations
    # interpolated to maintain monotonicity in adverse parent count.
    cpd14 = _cpt('14', ['14a', '14b', '14c', '14d'], [
        # P(N14 = Low distortion): 1 - P(High)
        [0.85, 0.70, 0.70, 0.50, 0.60, 0.45, 0.45, 0.25,
         0.55, 0.40, 0.40, 0.15, 0.35, 0.10, 0.15, 0.05],
        # P(N14 = High distortion) — anchored to §5.1.14 §7
        [0.15, 0.30, 0.30, 0.50, 0.40, 0.55, 0.55, 0.75,
         0.45, 0.60, 0.60, 0.85, 0.65, 0.90, 0.85, 0.95],
    ])

    # ── N15a: Tariff jurisdiction disparity (parent: N13 TraceRoute) ────────
    # Per Q4 (α): TraceRoute structural-bias signal conditions which provincial
    # tariff context the conviction was imposed in. When N13 (TraceRoute) is
    # High, the case engages structural-bias patterns that are themselves
    # correlated with high-tariff sentencing regions per Doob/Cesaroni/Roach.
    cpd15a = _cpt('15a', ['13'], [
        [0.65, 0.42],     # P(15a Low / Low-tariff) when N13 Low / High
        [0.35, 0.58],     # P(15a High / High-tariff) — rises with structural bias
    ])

    # ── N15b: Tariff-sensitive offence type (no parents — evidence) ─────────
    # Per Q6 (α): pattern-matched from offence text against tariff-sensitive
    # offence categories (Property, Drug, Administration of Justice).
    # Default prior reflects baseline rate.
    cpd15b = TabularCPD(variable='15b', variable_card=2, values=[[0.55], [0.45]])

    # ── N15c: Tariff-sensitive sentence length (no parents — evidence) ──────
    # Per Q6 (α): offence-conditional threshold:
    #   Tariff-sensitive offences: sentence > 1 year → state 1
    #   Conduct-driven offences (violent, sexual): sentence > 3 years → state 1
    # Default prior reflects baseline rate of long sentences.
    cpd15c = TabularCPD(variable='15c', variable_card=2, values=[[0.60], [0.40]])

    # ── N15d: Jurisprudential compliance absent (no parents — evidence) ─────
    # Per Q6 (α): inverse-attestation pattern. State 1 (doctrine absent) by
    # default; state 0 when counsel attests SCE/Tetrad applied at original
    # sentencing (shares n14c attestation — both are jurisprudence-applied
    # questions about the same sentencing event).
    # Per Q3 binary collapse: G+E (binding) and M+E (persuasive) both map to
    # state 0; only "None applied" maps to state 1.
    cpd15d = TabularCPD(variable='15d', variable_card=2, values=[[0.40], [0.60]])

    # ── N15: Interjurisdictional Tariff Distortion (5 parents) ──────────────
    # Per CH5 §5.1.15 §7 illustrative CPT (binary collapse, 4 anchors hit):
    #
    #   Parent state (N14, 15a, 15b, 15c, 15d) → P(N15=High distortion)
    #   col 18 (1,0,0,1,0) → 0.40   §7 row 5: LT+V+>3yr+M+E [binary collapse]
    #   col 19 (1,0,0,1,1) → 0.55   §7 row 6: LT+V+>3yr+None
    #   col 30 (1,1,1,1,0) → 0.65   §7 row 3: HT+Prop+>1yr+M+E
    #   col 31 (1,1,1,1,1) → 0.85   §7 row 1: HT+Prop+>1yr+None
    #
    # Big-endian column ordering per pgmpy:
    #   col_idx = state(N14)*16 + state(15a)*8 + state(15b)*4
    #             + state(15c)*2 + state(15d)*1
    #
    # Anchors hit at N14=1 (temporal distortion in play, per §5.1.15 Position
    # which places N15 downstream of N14). N14=0 columns are reduced versions
    # (~0.7-0.75× factor) modeling the temporal-correction effect.
    #
    # Binary collapse on G+E vs M+E: §5.1.15 §7 distinguishes binding (Gladue+
    # Ewert) from persuasive (Morris+Ellis) jurisprudence. Per Q3, binary
    # collapse maps both to "doctrine present" (15d=0). Anchored to M+E
    # values (more conservative — overestimating distortion is less harmful
    # than underestimating). G+E rows produce slightly stronger correction
    # but architecture treats them equivalently.
    cpd15 = _cpt('15', ['14', '15a', '15b', '15c', '15d'], [
        # P(N15 = Low distortion): 1 - P(High)
        [0.93, 0.80, 0.80, 0.60, 0.82, 0.70, 0.70, 0.50,
         0.80, 0.68, 0.70, 0.50, 0.68, 0.55, 0.50, 0.35,
         0.90, 0.72, 0.60, 0.45, 0.75, 0.58, 0.55, 0.35,
         0.70, 0.55, 0.55, 0.35, 0.55, 0.38, 0.35, 0.15],
        # P(N15 = High distortion) — anchored to §5.1.15 §7
        [0.07, 0.20, 0.20, 0.40, 0.18, 0.30, 0.30, 0.50,
         0.20, 0.32, 0.30, 0.50, 0.32, 0.45, 0.50, 0.65,
         0.10, 0.28, 0.40, 0.55, 0.25, 0.42, 0.45, 0.65,
         0.30, 0.45, 0.45, 0.65, 0.45, 0.62, 0.65, 0.85],
    ])

    # ── N16: Doctrinal Tension (parent: N10) ─────────────────────────────────
    # NEW node per CH5 §5.1.16 — s.718.04 vs s.718.2(e) conflict
    # Conservative default: tension is salient where SCE is misapplied
    # in cases involving Indigenous victims/offenders.
    cpd16 = _cpt('16', ['10'], [
        [0.75, 0.50],     # P(Low tension) — properly reconciled when SCE applied
        [0.25, 0.50],     # P(High tension) — unresolved conflict where SCE misapplied
    ])

    # ── N17a: Jurisdictional policing disparity (parent: N13 TraceRoute) ─────
    # Per CH5 §5.1.17 §4 + JP routing decision C3:
    # TraceRoute (structural systemic bias) conditions the disparity index.
    # When N13 (TraceRoute) is High, baseline disparity expectation rises.
    # Conservative defaults per JP M1 confirmation: Moderate when no structural
    # bias signal, elevated when N13 indicates systemic bias.
    cpd17a = _cpt('17a', ['13'], [
        [0.65, 0.40],     # P(disparity Low) — neutral when N13 Low; corrected when High
        [0.35, 0.60],     # P(disparity High) — rises with structural bias signal
    ])

    # ── N17b: Enforcement-disparity engagement (no parents — evidence) ───────
    # Per JP M3: derived from Gladue tab evidence (OR-gate) with counsel
    # attestation override. The Bayesian network treats this as a root evidence
    # node; the actual signal is computed app-side and fed at inference time.
    # Default prior: Low when no Gladue evidence engages §5.1.17 §2 categories.
    cpd17b = TabularCPD(variable='17b', variable_card=2, values=[[0.65], [0.35]])

    # ── N17c: Non-violent charge density (no parents — auto-computed) ────────
    # Per JP M2: pattern-matched from criminal record (breaches, AOJ offences,
    # possession). Default prior reflects baseline expectation across cases.
    cpd17c = TabularCPD(variable='17c', variable_card=2, values=[[0.65], [0.35]])

    # ── N17d: Surveillance-triggered entries (no parents — auto-computed) ────
    # Per JP M2: pattern-matched from criminal record (proactive enforcement
    # offence patterns). Default prior reflects baseline expectation.
    cpd17d = TabularCPD(variable='17d', variable_card=2, values=[[0.70], [0.30]])

    # ── N17: Over-policing & epistemic contamination (4 parents) ─────────────
    # Per CH5 §5.1.17 §7 illustrative CPT (anchors match exactly):
    #   (0,0,0,0) all Low                  → P(contamination) = 0.10
    #   (1,1,1,1) all High                 → P(contamination) = 0.85  [§7 row 1]
    #   (0,1,1,1) no disparity, race-eng+  → P(contamination) = 0.70  [§7 row 2]
    #   (1,0,1,1) high disp, no race-eng   → P(contamination) = 0.55  [§7 row 3]
    #
    # 16 parent combinations indexed in pgmpy column order:
    #   col_idx = state(17a) + state(17b)*2 + state(17c)*4 + state(17d)*8
    #
    # Structural weighting per §5.1.17 §5: disparity + race-engagement carry
    # more weight than record-derived signals alone (charge density, surveillance
    # entries). Single-parent contributions: N17a/N17b each ~0.30; N17c/N17d
    # each ~0.20. Pair contributions amplify (interaction effects). Anchor
    # points hit §5.1.17 §7 exactly; intermediate combinations interpolated
    # to maintain monotonicity in parent count.
    cpd17 = _cpt('17', ['17a', '17b', '17c', '17d'], [
        # P(N17 = Low contamination): 1 - P(High)
        [0.90, 0.80, 0.80, 0.70, 0.70, 0.45, 0.50, 0.30,
         0.70, 0.60, 0.60, 0.45, 0.50, 0.25, 0.30, 0.15],
        # P(N17 = High contamination) — anchored to §5.1.17 §7
        [0.10, 0.20, 0.20, 0.30, 0.30, 0.55, 0.50, 0.70,
         0.30, 0.40, 0.40, 0.55, 0.50, 0.75, 0.70, 0.85],
    ])

    # ── N18a: Jurisdiction SCE-integration sensitivity (parent: N13) ────────
    # Per Q2 (α) + Q5 (α): TraceRoute structural-bias signal conditions
    # whether the sentencing jurisdiction has strong provincial appellate
    # SCE-integration scrutiny (Morris ONCA 2021, Ellis BCCA 2022) versus
    # weaker provincial precedent. When N13 (TraceRoute) is High, structural-
    # bias patterns correlate with jurisdictions where SCE integration is
    # less likely to be substantively scrutinised.
    cpd18a = _cpt('18a', ['13'], [
        [0.62, 0.40],     # P(18a Low / Morris-Ellis jurisdiction) when N13 Low / High
        [0.38, 0.60],     # P(18a High / no strong provincial precedent)
    ])

    # ── N18b: SCE Presence in Reasons (no parents — evidence) ───────────────
    # Per Q1 (β) + Q7 (α): aggregate signal driven by per-conviction
    # SCE-integration tags (Full / Partial / Nominal / Absent) exposed in UI.
    # State 1 (SCE absent in reasons) when at least one conviction tagged
    # "Absent". Default prior reflects baseline incidence of records lacking
    # SCE reference in reasons (DOJ reports / Morris Audit empirical findings).
    cpd18b = TabularCPD(variable='18b', variable_card=2, values=[[0.45], [0.55]])

    # ── N18c: SCE Substance (no parents — evidence) ─────────────────────────
    # Per Q4 (α) + Morris Heuristic Audit: state 1 (nominal-only or absent)
    # when at least one conviction tagged "Nominal" or "Absent". Default
    # prior reflects Morris Audit finding that even where SCE is mentioned,
    # substantive integration is often absent or minimal.
    cpd18c = TabularCPD(variable='18c', variable_card=2, values=[[0.40], [0.60]])

    # ── N18d: Doctrinal Tagging compliance (parent: N10) ────────────────────
    # Per Q5 (α): kinship edge from N10 (Misapplication). When N10 indicates
    # judicial misapplication of SCE, doctrinal tagging is more likely
    # incomplete or error-flagged (state 1). Mirrors N14d and N17d patterns
    # where N10 → sub-node carries the misapplication signal into the relevant
    # sub-node compliance check.
    cpd18d = _cpt('18d', ['10'], [
        [0.60, 0.30],     # P(18d Low / Tagging accurate) when N10 Low / High
        [0.40, 0.70],     # P(18d High / Tagging incomplete or error)
    ])

    # ── N18: Gladue/Ewert/Morris/Ellis Profile (6 parents) ──────────────────
    # Per CH5 §5.1.18 §7 illustrative CPT (binary collapse, 3 anchors hit):
    #
    #   Parent state (N12, N14, 18a, 18b, 18c, 18d) → P(N18=Inflated)
    #   col 48 (1,1,0,0,0,0) → 0.10   §7 Row 1: Full Application + Accurate
    #   col 51 (1,1,0,0,1,1) → 0.60   §7 Row 2: Nominal Reference + Incomplete
    #   col 55 (1,1,0,1,1,1) → 0.90   §7 Row 3+4 collapsed: No Reference +
    #                                  Incomplete/Error (binary collapse uses
    #                                  Row 4 = 0.90, conservative — same
    #                                  convention as N15 G+E/M+E collapse)
    #
    # Big-endian column ordering per pgmpy:
    #   col_idx = state(N12)*32 + state(N14)*16 + state(18a)*8
    #             + state(18b)*4 + state(18c)*2 + state(18d)*1
    #
    # Anchors hit at N12=1, N14=1 (peak amplification) with 18a=0
    # (Morris/Ellis jurisdiction). N12=0 / N14=0 columns are reduced versions
    # modeling the absence of judicial-reasoning / temporal amplification.
    #
    # Binary collapse on §7 multivalent state space:
    #   18b (Presence): "Full" or "Nominal" → 0; "No Reference" → 1
    #   18c (Substance): "Full" → 0; "Nominal" or "No Reference" → 1
    #   18d (Tagging): "Accurate" → 0; "Incomplete" or "Error" → 1
    # The N18b/N18c split preserves the Morris Heuristic Audit insight that
    # nominal-only mention without substantive integration is a distinct
    # failure mode from outright absence.
    cpd18 = _cpt('18', ['12', '14', '18a', '18b', '18c', '18d'], [
        # P(N18 = Reliable) = 1 - P(Inflated)
        [0.95, 0.85, 0.80, 0.68, 0.85, 0.78, 0.65, 0.45,
         0.90, 0.80, 0.75, 0.60, 0.80, 0.68, 0.55, 0.35,
         0.93, 0.80, 0.75, 0.60, 0.80, 0.70, 0.55, 0.35,
         0.87, 0.73, 0.67, 0.50, 0.73, 0.62, 0.45, 0.25,
         0.92, 0.78, 0.72, 0.58, 0.78, 0.68, 0.52, 0.32,
         0.85, 0.70, 0.64, 0.48, 0.70, 0.58, 0.42, 0.22,
         0.90, 0.70, 0.60, 0.40, 0.70, 0.55, 0.35, 0.10,
         0.80, 0.60, 0.50, 0.30, 0.60, 0.45, 0.25, 0.05],
        # P(N18 = Inflated) — anchored to §5.1.18 §7
        [0.05, 0.15, 0.20, 0.32, 0.15, 0.22, 0.35, 0.55,
         0.10, 0.20, 0.25, 0.40, 0.20, 0.32, 0.45, 0.65,
         0.07, 0.20, 0.25, 0.40, 0.20, 0.30, 0.45, 0.65,
         0.13, 0.27, 0.33, 0.50, 0.27, 0.38, 0.55, 0.75,
         0.08, 0.22, 0.28, 0.42, 0.22, 0.32, 0.48, 0.68,
         0.15, 0.30, 0.36, 0.52, 0.30, 0.42, 0.58, 0.78,
         0.10, 0.30, 0.40, 0.60, 0.30, 0.45, 0.65, 0.90,
         0.20, 0.40, 0.50, 0.70, 0.40, 0.55, 0.75, 0.95],
    ])

    # ── N19: Collider Bias (parents: N14, N17) ───────────────────────────────
    # Maps from current N17 (Collider bias), CPT structure preserved
    cpd19 = _cpt('19', ['14', '17'], [
        [0.55, 0.40, 0.42, 0.28],
        [0.45, 0.60, 0.58, 0.72],
    ])

    # Add all CPDs (Node 20 excluded — computed post-VE)
    # §5.1.17 sub-nodes (17a/b/c/d) and §5.1.14 sub-nodes (14a/b/c/d) included.
    model.add_cpds(
        cpd1, cpd2, cpd3, cpd4, cpd5, cpd6, cpd7, cpd8,
        cpd9, cpd10, cpd11, cpd12, cpd13,
        cpd14a, cpd14b, cpd14c, cpd14d, cpd14,
        cpd15a, cpd15b, cpd15c, cpd15d, cpd15,
        cpd16, cpd17a, cpd17b, cpd17c, cpd17d, cpd17,
        cpd18a, cpd18b, cpd18c, cpd18d, cpd18,
        cpd19
    )

    assert model.check_model(), "Model CPDs are inconsistent — check tables."
    return model


def compute_do_risk(posteriors: dict) -> float:
    """
    Compute Node 20 (DO designation risk) post-VE using calibrated formula.

    Architecture per thesis (updated for CH5 canonical taxonomy):
    
    Step 1 — Record reliability multiplier (CH5 §5.1.7 + §RM.5/§RM.6):
      Where bail-denial cascade (N7) or IAC (N6) is High, violent history
      carries reduced evidentiary weight. record_reliability ∈ [0.40, 1.0]
    
    Step 2 — Tool validity (Ewert): N5 conditions weight on N3 risk-tool outputs
    
    Step 3 — Raw risk = weighted risk posteriors (N2, N3, N4) with discounts
    
    Step 4 — Distortion correction reduces DO risk: HIGH distortion REDUCES
      effective DO risk because it flags evidentiary contamination
    
    Step 5 — Age burnout multiplier from N14 (Temporal Distortion)
    
    References: Tolppanen Report (2018); Feeley (1979); R v Antic [2017] SCC 27
    """
    p = posteriors

    # Record reliability: bail-WCGP + IAC + N17 over-policing + N14 temporal
    # distortion all reduce violent history weight. Each captures a distinct
    # production-condition distortion mechanism that compromises record
    # reliability per §5.1.7, §5.1.6, §5.1.17, §5.1.14 respectively.
    #
    # Weights (per JP confirmations):
    #   N7 (bail cascade)        0.35  — strongest per §RM.5
    #   N6 (IAC)                 0.15  — moderate per §RM.6
    #   N17 (over-policing)      0.30  — per §5.1.17 Q2
    #   N14 (temporal distortion) 0.20  — per §5.1.14 Q4
    # Total max discount: 1.0 (when all four signals = 1.0)
    # Floor at 0.30 prevents complete collapse — record retains meaningful
    # weight even under extreme distortion (avoids divide-by-zero pathologies
    # in downstream weighted aggregation).
    record_reliability = float(np.clip(
        1.0
        - 0.35 * p.get(7, 0.5)    # N7  bail-WCGP cascade
        - 0.15 * p.get(6, 0.5)    # N6  IAC
        - 0.30 * p.get(17, 0.5)   # N17 over-policing
        - 0.20 * p.get(14, 0.5)   # N14 temporal distortion
        - 0.15 * p.get(18, 0.5),  # N18 SCE Profile audit (per §5.1.18 §6 + Q6=A)
        0.30, 1.0
    ))

    # Tool validity: N5 (invalid risk tools) discounts N3 (sexual offence profile)
    tool_validity = float(np.clip(
        1.0 - 0.45 * p.get(5, 0.5),
        0.30, 1.0
    ))

    # Raw risk: substantive risk nodes (N2, N3, N4) with appropriate discounts.
    # Per §5.1.18 §6 + Q6=A: N18 (SCE Profile audit) routes through
    # record_reliability (above) — old direct contribution at 0.25 weight in
    # raw was both magnitude-wrong (over-large) and direction-wrong (added to
    # raw instead of discounting). Now N18 properly discounts N2 through the
    # record_reliability multiplier, mirroring N14 and N17.
    raw = (
        0.30 * p.get(2, 0.5) * record_reliability +    # N2: discounted by record reliability
        0.25 * p.get(3, 0.5) * tool_validity +         # N3: discounted by Ewert (N5)
        0.20 * p.get(4, 0.5)                           # N4 dynamic risk
        # N18 EXCLUDED from raw — contributes via record_reliability per §5.1.18 §6
    )

    # Distortion: systemic-distortion-layer nodes downweight effective risk.
    # Updated per CH5 canonical taxonomy + §5.1.17 N17 + §5.1.14 N14 ops.
    # 
    # NOTE: N17 (over-policing), N14 (temporal distortion), and N18 (SCE
    # Profile audit) are all EXCLUDED from dst — they contribute via
    # record_reliability per §5.1.17 §6, §5.1.14 §6, and §5.1.18 §6 respectively.
    # Including them in both would double-count the production-condition
    # distortion signal. Weights previously assigned have been redistributed:
    #   N17 0.10 → +0.05 to N13, +0.05 to N10 (Stage 1 of N17 build)
    #   N14 0.06 → +0.04 to N13, +0.02 to N10 (per JP Q5)
    # Total redistribution: N13 gained +0.09; N10 gained +0.07.
    # This preserves total distortion-side weight at ~1.0 while routing both
    # N17 and N14 effects through record_reliability where §5.1.17 §6 and
    # §5.1.14 §6 doctrinally locate them ("Prior Record Reliability Modifier").
    dst = (
        0.18 * posteriors.get(5, 0.5) +    # N5  invalid risk tools
        0.12 * posteriors.get(6, 0.5) +    # N6  IAC
        0.08 * posteriors.get(7, 0.5) +    # N7  bail-WCGP cascade
        0.05 * posteriors.get(9, 0.5) +    # N9  IGT/treatment (mitigation)
        0.25 * posteriors.get(10, 0.5) +   # N10 SCE misapplication (+0.02 from N14)
        0.05 * posteriors.get(12, 0.5) +   # N12 judging-the-judge
        0.19 * posteriors.get(13, 0.5) +   # N13 TraceRoute (+0.04 from N14)
        0.04 * posteriors.get(15, 0.5) +   # N15 tariff distortion
        0.04 * posteriors.get(16, 0.5)     # N16 doctrinal tension
        # N14 EXCLUDED — contributes via record_reliability per §5.1.14 §6
        # N17 EXCLUDED — contributes via record_reliability per §5.1.17 §6
        # N19 (collider bias) intentionally excluded — its effect is on
        # the inference structure itself, not directly on the DO posterior
    )

    # Age burnout multiplier — N14 (Temporal Distortion) per CH5 §5.1.14
    # Schedule preserved from previous calibration:
    #   N14 < 0.65 → 1.00 (no correction)
    #   N14 = 0.85 → 0.66 (34% attenuation)
    #   N14 = 0.97 → 0.46 (54% attenuation)
    n14 = posteriors.get(14, 0.5)
    burnout_mult = float(np.clip(1.0 - 1.70 * max(0.0, n14 - 0.65), 0.35, 1.0))
    raw = raw * burnout_mult

    return float(np.clip(raw * (1.0 - 0.68 * dst) + 0.03, 0.05, 0.93))


def get_inference_engine(model):
    """Return a Variable Elimination inference engine."""
    return VariableElimination(model)


def query_do_risk(engine, evidence: dict) -> dict:
    """
    Run Variable Elimination for Nodes 1-19, then compute Node 20 post-VE.
    evidence: dict of {node_id_str: 0|1} for observed nodes.
    Returns dict of {node_id: P(High)} for all 20 nodes.
    """
    results = {}
    # Standard nodes 1-19 plus §5.1.17 (17a/b/c/d), §5.1.14 (14a/b/c/d),
    # §5.1.15 (15a/b/c/d), and §5.1.18 (18a/b/c/d) sub-nodes
    ve_nodes = ([str(i) for i in range(1, 20)]
                + ['17a', '17b', '17c', '17d']
                + ['14a', '14b', '14c', '14d']
                + ['15a', '15b', '15c', '15d']
                + ['18a', '18b', '18c', '18d'])

    for node in ve_nodes:
        # Sub-nodes (17a/17b/17c/17d) keyed by string; main nodes by int
        node_key = node if not node.isdigit() else int(node)
        if node in evidence:
            results[node_key] = float(evidence[node])
            continue
        try:
            q = engine.query(variables=[node], evidence=evidence, show_progress=False)
            results[node_key] = float(q.values[1])
        except Exception:
            results[node_key] = 0.5

    results[20] = compute_do_risk(results)
    return results


def get_default_priors() -> dict:
    """Return prior P(High) for each node (no evidence observed).
    
    CH5 canonical taxonomy. Values are illustrative anchors for the 
    constructive proof, not empirical calibrations.
    """
    return {
        1:  0.83,   # Burden of proof — BRD per Gardiner [1982] 2 SCR 368
        2:  0.65,   # Validated risk elevators
        3:  0.35,   # Sexual offence risk profile
        4:  0.55,   # Dynamic risk factor cluster
        5:  0.70,   # Current risk assessment tools
        6:  0.55,   # IAC
        7:  0.55,   # Bail-WCGP cascade
        8:  0.40,   # FASD
        9:  0.75,   # IGT and cultural treatment
        10: 0.60,   # SCE misapplication
        11: 0.25,   # Gaming risk detector
        12: 0.50,   # Judging the judge — judicial reasoning reliability
        13: 0.60,   # TraceRoute / Structural systemic bias
        14: 0.50,   # Temporal distortion
        15: 0.45,   # Tariff distortion
        16: 0.30,   # Doctrinal tension
        17: 0.65,   # Over-policing
        18: 0.40,   # SCE Profile audit
        19: 0.55,   # Collider bias
        20: 0.50,   # DO designation risk
        # §5.1.17 sub-nodes — defaults reflect conservative starting points
        "17a": 0.35,  # Jurisdictional policing disparity (M1: Moderate default)
        "17b": 0.35,  # Enforcement-disparity engagement (low until Gladue evidence)
        "17c": 0.35,  # Non-violent charge density (auto-computed from record)
        "17d": 0.30,  # Surveillance-triggered entries (auto-computed from record)
        # §5.1.14 sub-nodes — defaults reflect conservative starting points
        "14a": 0.35,  # Sentencing era severity (default Low/Moderate; year-driven)
        "14b": 0.35,  # Historical mandatory minimum (offence+year derived)
        "14c": 0.45,  # SCE absent at sentencing (year-derived; pre-2012 → absent)
        "14d": 0.40,  # Judicial competence absent (downstream of N10 misapp)
        # §5.1.15 sub-nodes — defaults reflect conservative starting points
        "15a": 0.35,  # Tariff jurisdiction (default LT until detected/attested)
        "15b": 0.45,  # Tariff-sensitive offence (offence-text derived)
        "15c": 0.40,  # Tariff-sensitive sentence length (sentence-type derived)
        "15d": 0.55,  # Doctrine absent (default absent until SCE attested)
        # §5.1.18 sub-nodes — defaults reflect conservative starting points
        # consistent with Morris Heuristic Audit empirical findings
        "18a": 0.45,  # Jurisdiction sensitivity (default no Morris/Ellis until detected)
        "18b": 0.55,  # SCE presence in reasons (default absent until tagged)
        "18c": 0.60,  # SCE substance (default nominal-or-absent — Morris Audit)
        "18d": 0.50,  # Doctrinal tagging compliance (default Incomplete; downstream of N10)
    }
