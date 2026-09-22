from app.services.decision import DecisionService


def test_decision_service_applies_three_level_risk():
    service = DecisionService(threshold_low=0.3, threshold_high=0.7)
    assert service.decide(0.10).decision == "APPROVE"
    assert service.decide(0.50).decision == "REVIEW"
    assert service.decide(0.90).decision == "BLOCK"
