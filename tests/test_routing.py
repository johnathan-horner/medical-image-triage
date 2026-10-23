import numpy as np

from triage.config import ROUTE_AUTO, ROUTE_REVIEW
from triage.routing import confidence_of, pick_threshold, route, routing_report, stratified_halves


def test_route_boundary():
    assert route(0.90, 0.90) == ROUTE_AUTO
    assert route(0.8999, 0.90) == ROUTE_REVIEW


def test_confidence_is_symmetric():
    np.testing.assert_allclose(confidence_of(np.array([0.1, 0.5, 0.95])), [0.9, 0.5, 0.95])


def test_pick_threshold_excludes_uncertain_errors():
    # Confident cases are right, uncertain ones (p near 0.5) are wrong.
    prob = np.array([0.99] * 50 + [0.01] * 50 + [0.55] * 10 + [0.45] * 10)
    labels = np.array([1] * 50 + [0] * 50 + [0] * 10 + [1] * 10)
    t = pick_threshold(prob, labels, target_accuracy=0.98)
    assert 0.55 < t <= 0.99
    rep = routing_report(prob, labels, t)
    assert rep["auto_triage"]["accuracy"] == 1.0
    assert rep["human_review"]["cases"] == 20


def test_pick_threshold_unreachable_sends_everything_to_review():
    prob = np.full(100, 0.99)
    labels = np.zeros(100, dtype=int)
    assert pick_threshold(prob, labels) == 1.01
    assert routing_report(prob, labels, 1.01)["auto_triage"]["cases"] == 0


def test_missed_pneumonia_counted_per_lane():
    prob = np.array([0.02, 0.4, 0.9])
    labels = np.array([1, 1, 1])
    rep = routing_report(prob, labels, 0.9)
    assert rep["auto_triage"]["missed_pneumonia"] == 1
    assert rep["human_review"]["missed_pneumonia"] == 1


def test_stratified_halves_disjoint_and_balanced():
    labels = np.array([0] * 40 + [1] * 60)
    a, b = stratified_halves(labels, seed=0)
    assert len(set(a) & set(b)) == 0
    assert len(a) + len(b) == 100
    assert labels[a].sum() == 30 and labels[b].sum() == 30
    np.testing.assert_array_equal(a, stratified_halves(labels, seed=0)[0])
