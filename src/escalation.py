"""Graded alert escalation and a *simulated* autopilot handover.

This module sits above :mod:`src.detector`. It takes the per-frame driver
state (ALERT / TIRED / MICROSLEEP, or ``None`` when no face is visible) and
decides what the car should do:

    NONE      normal manual driving
    WARNING   driver looks tired for a sustained period -> visual + soft beep
    PROMPT    severe or persistent drowsiness -> "press SPACE if awake"
    TAKEOVER  no response to the prompt -> autopilot engages (SIMULATED)

Like the rest of ``src/`` it has no OpenCV / MediaPipe imports and takes the
time as an argument, so every rule can be unit tested without a camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

from .detector import State


class Level(IntEnum):
    """Escalation level. Higher number = more serious."""

    NONE = 0
    WARNING = 1
    PROMPT = 2
    TAKEOVER = 3


@dataclass
class EscalationConfig:
    """Timing rules, all in seconds."""

    tired_seconds: float = 3.0              # sustained TIRED before WARNING
    warning_to_prompt_seconds: float = 8.0  # still impaired at WARNING -> PROMPT
    prompt_timeout_seconds: float = 5.0     # no acknowledgement -> TAKEOVER
    recover_seconds: float = 3.0            # continuous ALERT to clear a WARNING
    no_face_seconds: float = 4.0            # face lost this long -> treated as severe
    ack_grace_seconds: float = 3.0          # after an acknowledgement, no new PROMPT yet


@dataclass
class Event:
    """One level change, kept for the log and the report."""

    time: float
    old: Level
    new: Level
    reason: str


class Escalation:
    """Stateful escalation logic. Call :meth:`update` once per frame."""

    def __init__(self, config: Optional[EscalationConfig] = None) -> None:
        self.config = config or EscalationConfig()
        self.level = Level.NONE
        self.events: List[Event] = []
        self._level_since = 0.0
        self._impaired_since: Optional[float] = None
        self._alert_since: Optional[float] = None
        self._no_face_since: Optional[float] = None
        self._grace_until = -1.0

    # -- helpers ---------------------------------------------------------
    def _set(self, new: Level, now: float, reason: str) -> None:
        if new == self.level:
            return
        self.events.append(Event(now, self.level, new, reason))
        self.level = new
        self._level_since = now

    def prompt_remaining(self, now: float) -> float:
        """Seconds left to acknowledge (0 unless a prompt is showing)."""
        if self.level != Level.PROMPT:
            return 0.0
        return max(0.0, self.config.prompt_timeout_seconds - (now - self._level_since))

    # -- main step -------------------------------------------------------
    def update(self, state: Optional[State], now: float, acknowledged: bool = False) -> Level:
        """Advance the logic by one frame and return the current level.

        ``state`` is ``None`` when no face is visible. ``acknowledged`` is True
        on the frame the driver presses the acknowledge key.
        """
        cfg = self.config

        # Once the autopilot has taken over it stays engaged until resume().
        if self.level == Level.TAKEOVER:
            return self.level

        # --- face lost -------------------------------------------------
        if state is None:
            if self._no_face_since is None:
                self._no_face_since = now
            severe = (now - self._no_face_since) >= cfg.no_face_seconds
            if not severe:
                # Short dropout (head turn, tracking glitch): freeze the timers.
                return self._prompt_logic(now, acknowledged)
            tired = False
            reason_severe = "face lost"
        else:
            self._no_face_since = None
            severe = state == State.MICROSLEEP
            tired = state == State.TIRED
            reason_severe = "microsleep"

        impaired = severe or tired
        if impaired:
            self._alert_since = None
            if self._impaired_since is None:
                self._impaired_since = now
        else:
            self._impaired_since = None
            if self._alert_since is None:
                self._alert_since = now

        if self.level == Level.PROMPT:
            return self._prompt_logic(now, acknowledged)

        in_grace = now < self._grace_until

        if self.level == Level.NONE:
            if severe and not in_grace:
                self._set(Level.PROMPT, now, f"{reason_severe}: immediate prompt")
            elif impaired and now - self._impaired_since >= cfg.tired_seconds:
                self._set(Level.WARNING, now, "sustained tiredness")
        elif self.level == Level.WARNING:
            if severe and not in_grace:
                self._set(Level.PROMPT, now, f"{reason_severe} during warning")
            elif (
                impaired
                and not in_grace
                and now - self._level_since >= cfg.warning_to_prompt_seconds
            ):
                self._set(Level.PROMPT, now, "tiredness persisted after warning")
            elif not impaired and now - self._alert_since >= cfg.recover_seconds:
                self._set(Level.NONE, now, "driver alert again")
        return self.level

    def _prompt_logic(self, now: float, acknowledged: bool) -> Level:
        if self.level != Level.PROMPT:
            return self.level
        if acknowledged:
            self._grace_until = now + self.config.ack_grace_seconds
            self._set(Level.WARNING, now, "driver acknowledged")
        elif now - self._level_since >= self.config.prompt_timeout_seconds:
            self._set(Level.TAKEOVER, now, "no response to prompt: autopilot engaged")
        return self.level

    # -- hand back ---------------------------------------------------------
    def resume(self, state: Optional[State], now: float) -> bool:
        """Hand control back to the driver. Only allowed if they look ALERT."""
        if self.level != Level.TAKEOVER:
            return False
        if state != State.ALERT:
            return False
        self._set(Level.NONE, now, "driver resumed manual control")
        self._impaired_since = None
        self._alert_since = now
        self._no_face_since = None
        return True


@dataclass
class VehicleSim:
    """Toy vehicle: cruises at a set speed, brakes smoothly under TAKEOVER.

    This is the *simulated* part of the handover. It does not touch a real
    vehicle; it only shows what a controlled slow-down would look like.
    """

    cruise_kmh: float = 80.0
    brake_kmh_per_s: float = 8.0
    speed_kmh: float = field(default=80.0)

    def step(self, dt: float, level: Level) -> float:
        if level == Level.TAKEOVER:
            self.speed_kmh = max(0.0, self.speed_kmh - self.brake_kmh_per_s * dt)
        elif self.speed_kmh < self.cruise_kmh and self.speed_kmh > 0.0:
            # Driver took control back: they resume cruising.
            self.speed_kmh = min(self.cruise_kmh, self.speed_kmh + 10.0 * dt)
        return self.speed_kmh

    @property
    def stopped(self) -> bool:
        return self.speed_kmh <= 0.0
