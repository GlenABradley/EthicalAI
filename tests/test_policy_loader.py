from ethicalai.interaction.policy import load_policy

def test_policy_loader_defaults():
    p = load_policy()
    assert p.strictness in {"permissive","balanced","strict","paranoid"}
    # Effective τ scales
    assert p.effective_tau(1.0) > 0
