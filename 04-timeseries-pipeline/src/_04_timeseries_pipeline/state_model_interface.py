from state_model.filtering import StateModel

model = StateModel.load("models/omen_v1.npz")


def on_pipeline_packet(packet: dict) -> dict:
    # Called once per second by the Time-Series Pipeline.
    result = model.update(packet)
    return {
        "current_state": result["current_state"],  # HMM candidate gated by FSM
        "previous_state": result["previous_state"],
        "time_in_state": result["time_in_state"],
        "state_probabilities": result["state_probabilities"],
        "transition_history": result["transition_history"],
    }