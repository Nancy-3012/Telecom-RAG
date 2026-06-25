# Hand-written RAN troubleshooting knowledge -- fills a real gap TeleQnA has
# for the root-cause-analysis use case. TeleQnA is excellent for spec Q&A
# (it's an exam-style question bank), but it doesn't connect facts into
# causal explanations the way a troubleshooting guide would. These entries
# do that specifically for the kind of telemetry anomalies anomaly_detector.py
# flags (BLER, RSRP, SNR issues).

TROUBLESHOOTING_FACTS = [
    {
        "id": "kb_001",
        "text": "A consistently high downlink block error rate (DL BLER) generally "
                "indicates poor radio link quality. Common causes include low "
                "signal-to-noise ratio (SNR), severe interference from neighboring "
                "cells, multipath fading, or the UE being located near the cell "
                "edge with high path loss.",
        "source": "Troubleshooting KB",
    },
    {
        "id": "kb_002",
        "text": "A DL BLER value at or near 100% over a sustained period typically "
                "indicates a complete radio link failure rather than degraded but "
                "functioning transmission. This can result from the UE losing "
                "synchronization, moving out of coverage, severe blocking or "
                "shadowing, or a hardware fault at the serving cell.",
        "source": "Troubleshooting KB",
    },
    {
        "id": "kb_003",
        "text": "Low RSRP (Reference Signal Received Power) indicates the UE is "
                "receiving a weak signal from the serving cell, commonly caused by "
                "large distance from the cell, physical obstructions, poor antenna "
                "alignment, or high building or terrain attenuation.",
        "source": "Troubleshooting KB",
    },
    {
        "id": "kb_004",
        "text": "Radio Link Failure (RLF) in cellular networks is commonly caused by "
                "handover failures, sudden coverage holes, excessive interference, "
                "or the UE moving faster than the network's mobility management "
                "procedures can track.",
        "source": "Troubleshooting KB",
    },
    {
        "id": "kb_005",
        "text": "A sudden spike in BLER without a corresponding drop in RSRP may "
                "point to interference, scheduling conflicts, or hardware issues at "
                "the base station rather than a pure coverage problem, since the "
                "received signal strength itself remains adequate.",
        "source": "Troubleshooting KB",
    },
    {
        "id": "kb_006",
        "text": "Hardware impairments at the transmitter or receiver, such as power "
                "amplifier nonlinearity or oscillator phase noise, increase "
                "effective noise power and can degrade the effective SINR, which "
                "in turn increases the block error rate even when nominal signal "
                "strength appears normal.",
        "source": "Troubleshooting KB",
    },
    {
        "id": "kb_007",
        "text": "Network congestion or excessive scheduling load on a cell can lead "
                "to increased retransmissions and higher observed block error "
                "rates during peak traffic periods, even without any change in "
                "radio propagation conditions.",
        "source": "Troubleshooting KB",
    },
    {
        "id": "kb_008",
        "text": "A UE transitioning between RRC_CONNECTED and RRC_IDLE states, or "
                "undergoing a failed handover attempt, may show telemetry "
                "indicating zero throughput and maximum block error rate during "
                "the transition window. This can resemble a persistent anomaly "
                "but is often transient rather than indicating a hardware fault.",
        "source": "Troubleshooting KB",
    },
]