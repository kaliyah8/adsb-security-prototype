from __future__ import annotations

ANOMALY_PLAYBOOK = {
    "Normal Flight": {
        "title": "Normal Flight Behavior",
        "summary": (
            "This is the baseline case. The aircraft moves smoothly, speed changes are gradual, "
            "and reported position agrees with believable motion over time."
        ),
        "why_it_matters": (
            "A good monitoring system must understand normal behavior first. Normal flight is the reference used to detect suspicious behavior."
        ),
        "what_you_see": [
            "A smooth path on the map",
            "Gradual changes in speed and direction",
            "Small mismatch between reported speed and motion-implied speed",
            "Few or no warnings",
        ],
        "detector_cues": [
            "Reasonable distance traveled per update",
            "Stable acceleration and turn rate",
            "Good data quality score",
        ],
    },
    "Teleportation": {
        "title": "Teleportation / Impossible Jump",
        "summary": (
            "The aircraft suddenly appears far away from its previous position, in a way that is not physically possible for the elapsed time."
        ),
        "why_it_matters": (
            "A false jump can distort situational awareness, create confusion, and mislead an operator about where an aircraft really is."
        ),
        "what_you_see": [
            "A sharp break in the path",
            "A sudden jump from one location to another",
            "Implied speed spikes far above normal",
            "Alert changes immediately during the jump window",
        ],
        "detector_cues": [
            "Distance moved is too large for the time gap",
            "Implied speed is unrealistically high",
            "Reported speed and position-derived speed disagree strongly",
        ],
    },
    "GPS Spoofing": {
        "title": "GPS Spoofing / Position Mismatch",
        "summary": (
            "The aircraft position is shifted or drifted from where it should be. The track may still look smooth, but it no longer matches believable motion."
        ),
        "why_it_matters": (
            "This can displace a real aircraft without creating one obvious giant jump, making it harder to notice."
        ),
        "what_you_see": [
            "A drifted or offset path",
            "Motion that looks somewhat smooth but mathematically inconsistent",
            "Growing disagreement between reported speed and position changes",
        ],
        "detector_cues": [
            "Increasing speed mismatch",
            "Position updates that do not fit recent motion history",
            "Possible drops in quality score near suspicious points",
        ],
    },
    "Ghost Aircraft": {
        "title": "Ghost Aircraft / Fake Track",
        "summary": (
            "A false or synthetic aircraft track appears. It may move in unrealistic ways or behave unlike a believable aircraft."
        ),
        "why_it_matters": (
            "Ghost targets waste attention, create false alarms, and can pollute the airspace picture."
        ),
        "what_you_see": [
            "A track that seems artificial or unsupported by believable history",
            "Unexpected motion patterns",
            "Suspicious turning or acceleration behavior",
        ],
        "detector_cues": [
            "Unusual acceleration or turn rate",
            "Motion that does not resemble normal aircraft dynamics",
            "Behavior that stands out from surrounding traffic",
        ],
    },
}