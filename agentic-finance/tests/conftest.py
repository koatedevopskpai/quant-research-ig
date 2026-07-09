import pytest
import numpy as np
from agentic_finance.message_bus import MessageBus
from agentic_finance.audit_trail import AuditTrail


@pytest.fixture
def message_bus():
    return MessageBus()


@pytest.fixture
def audit_trail():
    return AuditTrail()


@pytest.fixture
def sample_returns():
    return np.random.normal(0.001, 0.02, 500).tolist()
