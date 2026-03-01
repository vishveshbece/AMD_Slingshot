"""
state_machine.py — Lab Experiment State Machine
================================================
Tracks a 3-step classroom lab: Setup → Processing → Verification.
Uses the `transitions` library (pip install transitions).

Key behaviours
--------------
* on_enter_<state> callbacks log entry and emit a UI event.
* on_skip_<step>   callbacks fire when a transition is triggered out-of-order,
                   delivering a contextual hint to the student.
* Thread-safe: all public methods acquire a lock before mutating state.
"""

import logging
import threading
from datetime import datetime
from typing import Callable, Optional

from transitions import Machine, MachineError

# ---------------------------------------------------------------------------
# Configure module-level logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
log = logging.getLogger("LabStateMachine")


# ---------------------------------------------------------------------------
# Hint catalogue — indexed by (from_state, to_state) tuples
# ---------------------------------------------------------------------------
HINTS: dict[tuple[str, str], str] = {
    ("setup",      "verification"): (
        "⚠️  You jumped straight to Verification! "
        "Complete the Processing step first — ensure the kernel is loaded "
        "and at least one inference has been run."
    ),
    ("idle",       "processing"): (
        "⚠️  Setup hasn't been completed yet. "
        "Please connect the camera and load the DPU overlay before processing."
    ),
    ("idle",       "verification"): (
        "⚠️  Neither Setup nor Processing is complete. "
        "Follow the lab steps in order for valid results."
    ),
}


# ---------------------------------------------------------------------------
# LabStateMachine
# ---------------------------------------------------------------------------
class LabStateMachine:
    """
    State diagram
    ─────────────
        ┌──────┐   start_setup     ┌───────┐   begin_processing  ┌────────────┐
        │ idle │──────────────────▶│ setup │────────────────────▶│ processing │
        └──────┘                   └───────┘                      └────────────┘
                                        │  skip_to_verify (hint)       │
                                        └──────────────────────────────┤
                                                                        │ verify
                                                                        ▼
                                                                  ┌────────────┐
                                                                  │verification│
                                                                  └────────────┘
                                                                        │ reset
                                                                        ▼
                                                                     idle
    """

    STATES = ["idle", "setup", "processing", "verification"]

    TRANSITIONS = [
        # Normal happy path
        {"trigger": "start_setup",        "source": "idle",        "dest": "setup"},
        {"trigger": "begin_processing",   "source": "setup",       "dest": "processing"},
        {"trigger": "verify",             "source": "processing",  "dest": "verification"},
        {"trigger": "reset",              "source": "*",           "dest": "idle"},

        # Skip transitions — dest still advances but callbacks fire hints
        {
            "trigger":    "skip_to_verify",
            "source":     ["idle", "setup"],
            "dest":       "verification",
            "before":     "_hint_on_skip",      # fires BEFORE the state changes
        },
        {
            "trigger":    "force_processing",
            "source":     "idle",
            "dest":       "processing",
            "before":     "_hint_on_skip",
        },
    ]

    def __init__(self, hint_callback: Optional[Callable[[str], None]] = None):
        """
        Parameters
        ----------
        hint_callback : callable(str) | None
            Function to call with the hint message string whenever a step is
            skipped. If None, the hint is only logged.
        """
        self.state: str = "idle"
        self._lock = threading.Lock()
        self._hint_callback = hint_callback or self._default_hint_handler
        self._history: list[dict] = []   # audit log of state transitions

        self._machine = Machine(
            model=self,
            states=self.STATES,
            transitions=self.TRANSITIONS,
            initial="idle",
            send_event=True,           # passes EventData to callbacks
            ignore_invalid_triggers=False,
        )

    # ------------------------------------------------------------------
    # on_enter_* callbacks — called automatically by transitions library
    # ------------------------------------------------------------------
    def on_enter_idle(self, event):
        log.info("🔄  Lab reset — returning to IDLE.")
        self._record("idle", event)

    def on_enter_setup(self, event):
        log.info("🔧  SETUP started — camera and DPU overlay initialisation.")
        self._record("setup", event)

    def on_enter_processing(self, event):
        log.info("⚙️   PROCESSING — HLS kernel active, running inference.")
        self._record("processing", event)

    def on_enter_verification(self, event):
        log.info("✅  VERIFICATION — comparing DPU outputs against expected values.")
        self._record("verification", event)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _hint_on_skip(self, event):
        """Before-callback for all skip transitions."""
        from_state = event.transition.source
        to_state   = event.transition.dest
        key        = (from_state, to_state)
        hint       = HINTS.get(key, f"⚠️  Skipped from '{from_state}' to '{to_state}'.")
        log.warning(hint)
        self._hint_callback(hint)

    @staticmethod
    def _default_hint_handler(hint: str):
        """Fallback — prints hint to stdout if no callback was supplied."""
        print(f"\n{'='*60}\nHINT FOR STUDENT:\n{hint}\n{'='*60}\n")

    def _record(self, new_state: str, event):
        entry = {
            "timestamp":  datetime.utcnow().isoformat(),
            "state":      new_state,
            "trigger":    event.event.name,
        }
        self._history.append(entry)

    # ------------------------------------------------------------------
    # Thread-safe public API
    # ------------------------------------------------------------------
    def advance(self, trigger: str) -> bool:
        """
        Fire a trigger by name in a thread-safe manner.

        Returns True on success, False if the trigger is invalid for the
        current state (logged as a warning rather than raising).
        """
        with self._lock:
            try:
                getattr(self, trigger)()
                return True
            except (MachineError, AttributeError) as exc:
                log.warning("Trigger '%s' rejected in state '%s': %s",
                            trigger, self.state, exc)
                return False

    @property
    def current_step(self) -> str:
        """Human-readable current step label for the GUI."""
        labels = {
            "idle":         "Not Started",
            "setup":        "Step 1 — Setup",
            "processing":   "Step 2 — Processing",
            "verification": "Step 3 — Verification",
        }
        return labels.get(self.state, self.state.title())

    @property
    def progress_pct(self) -> int:
        """0–100 progress percentage for a progress bar widget."""
        return {"idle": 0, "setup": 33, "processing": 66, "verification": 100}.get(
            self.state, 0
        )

    def get_history(self) -> list[dict]:
        """Return a copy of the transition audit log."""
        with self._lock:
            return list(self._history)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def my_hint(msg: str):
        print(f"[GUI ALERT] {msg}")

    sm = LabStateMachine(hint_callback=my_hint)
    print(f"Initial : {sm.current_step}")

    sm.advance("start_setup")
    print(f"After setup   : {sm.current_step}  ({sm.progress_pct}%)")

    # Simulate a student skipping Processing
    sm.advance("skip_to_verify")
    print(f"After skip    : {sm.current_step}  ({sm.progress_pct}%)")

    sm.advance("reset")
    print(f"After reset   : {sm.current_step}  ({sm.progress_pct}%)")