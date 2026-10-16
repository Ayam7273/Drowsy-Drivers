"""Unit tests for the escalation logic (no camera needed)."""

import unittest

from src.detector import State
from src.escalation import Escalation, EscalationConfig, Level, VehicleSim


def run(esc, state, start, seconds, dt=0.1, ack_at=None):
    """Feed ``state`` for ``seconds`` and return the time reached."""
    t = start
    end = start + seconds
    while t < end - 1e-9:
        esc.update(state, t, acknowledged=(ack_at is not None and abs(t - ack_at) < dt / 2))
        t += dt
    return t


class EscalationTests(unittest.TestCase):
    def test_alert_driver_never_escalates(self):
        esc = Escalation()
        run(esc, State.ALERT, 0, 60)
        self.assertEqual(esc.level, Level.NONE)
        self.assertEqual(esc.events, [])

    def test_brief_tiredness_is_ignored(self):
        esc = Escalation()
        t = run(esc, State.TIRED, 0, 2.0)
        run(esc, State.ALERT, t, 5.0)
        self.assertEqual(esc.level, Level.NONE)

    def test_sustained_tiredness_gives_warning(self):
        esc = Escalation()
        run(esc, State.TIRED, 0, 3.5)
        self.assertEqual(esc.level, Level.WARNING)

    def test_warning_clears_after_recovery(self):
        esc = Escalation()
        t = run(esc, State.TIRED, 0, 3.5)
        run(esc, State.ALERT, t, 3.5)
        self.assertEqual(esc.level, Level.NONE)

    def test_persistent_tiredness_escalates_to_prompt(self):
        esc = Escalation()
        run(esc, State.TIRED, 0, 3.2 + 8.2)
        self.assertEqual(esc.level, Level.PROMPT)

    def test_microsleep_prompts_immediately(self):
        esc = Escalation()
        esc.update(State.MICROSLEEP, 0.0)
        self.assertEqual(esc.level, Level.PROMPT)

    def test_no_response_engages_autopilot(self):
        esc = Escalation()
        esc.update(State.MICROSLEEP, 0.0)
        run(esc, State.MICROSLEEP, 0.1, 5.2)
        self.assertEqual(esc.level, Level.TAKEOVER)

    def test_acknowledge_drops_to_warning(self):
        esc = Escalation()
        esc.update(State.MICROSLEEP, 0.0)
        esc.update(State.ALERT, 1.0, acknowledged=True)
        self.assertEqual(esc.level, Level.WARNING)

    def test_grace_period_after_acknowledge(self):
        esc = Escalation()
        esc.update(State.MICROSLEEP, 0.0)
        esc.update(State.ALERT, 1.0, acknowledged=True)
        esc.update(State.MICROSLEEP, 1.5)  # inside grace
        self.assertEqual(esc.level, Level.WARNING)
        esc.update(State.MICROSLEEP, 4.5)  # grace over
        self.assertEqual(esc.level, Level.PROMPT)

    def test_takeover_is_latched(self):
        esc = Escalation()
        esc.update(State.MICROSLEEP, 0.0)
        run(esc, State.ALERT, 0.1, 10.0)
        self.assertEqual(esc.level, Level.TAKEOVER)

    def test_resume_requires_alert_driver(self):
        esc = Escalation()
        esc.update(State.MICROSLEEP, 0.0)
        run(esc, State.MICROSLEEP, 0.1, 5.2)
        self.assertFalse(esc.resume(State.TIRED, 10.0))
        self.assertFalse(esc.resume(None, 10.0))
        self.assertEqual(esc.level, Level.TAKEOVER)
        self.assertTrue(esc.resume(State.ALERT, 10.0))
        self.assertEqual(esc.level, Level.NONE)

    def test_resume_without_takeover_does_nothing(self):
        self.assertFalse(Escalation().resume(State.ALERT, 0.0))

    def test_short_face_dropout_does_not_escalate(self):
        esc = Escalation()
        run(esc, State.ALERT, 0, 2.0)
        run(esc, None, 2.0, 2.0)
        self.assertEqual(esc.level, Level.NONE)

    def test_long_face_loss_prompts(self):
        esc = Escalation()
        run(esc, None, 0, 4.5)
        self.assertEqual(esc.level, Level.PROMPT)

    def test_prompt_remaining_counts_down(self):
        esc = Escalation(EscalationConfig(prompt_timeout_seconds=5.0))
        esc.update(State.MICROSLEEP, 0.0)
        self.assertAlmostEqual(esc.prompt_remaining(2.0), 3.0)
        self.assertEqual(Escalation().prompt_remaining(1.0), 0.0)

    def test_events_record_reasons(self):
        esc = Escalation()
        esc.update(State.MICROSLEEP, 0.0)
        self.assertEqual(len(esc.events), 1)
        self.assertEqual(esc.events[0].new, Level.PROMPT)


class VehicleSimTests(unittest.TestCase):
    def test_brakes_to_stop_under_takeover(self):
        v = VehicleSim()
        for _ in range(200):
            v.step(0.1, Level.TAKEOVER)
        self.assertTrue(v.stopped)

    def test_holds_speed_when_manual(self):
        v = VehicleSim()
        v.step(1.0, Level.NONE)
        self.assertEqual(v.speed_kmh, 80.0)


if __name__ == "__main__":
    unittest.main()
