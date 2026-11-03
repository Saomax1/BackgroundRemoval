"""Help > Check for Updates, the update dialog, and finishing a fresh install.

``install(window)`` adds the menu items; ``on_startup(window)`` runs once after
the window shows. It reports how the last update went, offers to download the
parts Setup.exe leaves out (GPU support and the AI models) on the first start,
and, only if the user turned it on, quietly looks for a newer release.
The work itself lives in ``cutout.update.client``.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

from ..update import client
from ..update.manifest import RELEASES_PAGE, Manifest

if TYPE_CHECKING:  # pragma: no cover
    from .main_window import MainWindow

log = logging.getLogger(__name__)

STARTUP_CHECK_KEY = "updates/check_at_startup"


def _size(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.1f} GB"
    return f"{n / 1e6:.0f} MB" if n >= 10e6 else f"{n / 1e6:.1f} MB"


def _flag(value) -> bool:
    return value in (True, "true", "1", 1)


def install(window: "MainWindow") -> None:
    m = getattr(window, "help_menu", None) or window.menuBar().addMenu("&Help")
    m.addSeparator()
    act_check = QAction("Check for Updates…", window)
    act_check.setStatusTip("Look on GitHub for a newer version of Cutout (needs the internet)")
    act_check.triggered.connect(lambda: check_now(window, manual=True))
    m.addAction(act_check)

    act_auto = QAction("Check for Updates at Startup", window)
    act_auto.setCheckable(True)
    act_auto.setChecked(_flag(window.settings.value(STARTUP_CHECK_KEY, False)))
    act_auto.setStatusTip("When on, Cutout looks for a new version each time it opens, if online")
    act_auto.toggled.connect(lambda on: window.settings.setValue(STARTUP_CHECK_KEY, on))
    m.addAction(act_auto)

    act_about = QAction("About Cutout", window)
    act_about.triggered.connect(lambda: show_about(window))
    m.addAction(act_about)

    window.act_check_updates = act_check
    window.act_auto_updates = act_auto
    window.update_status = None  # filled in by on_startup for the installed app


def show_about(window) -> None:
    from .. import __version__

    status = getattr(window, "update_status", None)
    version = status.version if status else __version__
    text = f"Cutout {version}\n\nRemoves backgrounds from photos and anime art, fully offline."
    if getattr(window, "ask_user", False):
        QMessageBox.about(window, "About Cutout", text)
    else:
        window.statusBar().showMessage(text.split("\n")[0], 4000)


# ------------------------------------------------------------- startup


def on_startup(window: "MainWindow") -> None:
    result = client.take_last_result()
    if result is not None:
        if result.get("ok"):
            window.statusBar().showMessage(f"Cutout was updated to version {result.get('version')}.", 10000)
            log.info("update to %s applied", result.get("version"))
        else:
            log.error("update failed: %s (log: %s)", result.get("message"), result.get("log"))
            _warn(window, "Update not installed", result.get("message") or "The update could not be installed.")
    if client.app_dir() is None:
        return  # running from source: nothing to finish or update

    def done(status):
        window.update_status = status
        if status is None:
            return
        if status.missing:
            offer_finish_setup(window, status)
        elif _flag(window.settings.value(STARTUP_CHECK_KEY, False)):
            check_now(window, manual=False)

    window.run_in_background(client.local_status, on_done=done,
                             on_error=lambda e: log.warning("cannot read install state: %s", e))


def offer_finish_setup(window, status: client.Status) -> None:
    plan = client.plan_update(status, status.release)
    if not plan.fetch:
        return
    dlg = UpdateDialog(window, status, plan, first_run=True)
    window.update_dialog = dlg
    dlg.show()


# --------------------------------------------------------------- check


def check_now(window, manual: bool) -> None:
    status = getattr(window, "update_status", None)
    if status is None:
        if client.app_dir() is None:
            if manual:
                _info(window, "Updates", "Updates work in the installed app. This copy runs from source code.")
            return
        # The startup scan has not finished yet: read it now, then check.
        window.run_in_background(client.local_status, on_done=lambda s: _check_with(window, s, manual),
                                 on_error=lambda e: manual and _warn(window, "Updates", str(e)))
        return
    _check_with(window, status, manual)


def _check_with(window, status, manual: bool) -> None:
    window.update_status = status
    if status is None:
        if manual:
            _info(window, "Updates", "This copy of Cutout cannot update itself. Install it with the Setup "
                                     "from the releases page once, and later updates work from here.")
        return
    if manual:
        window.statusBar().showMessage("Checking for updates…")

    def got(target: Manifest):
        window.statusBar().clearMessage()
        plan = client.plan_update(status, target)
        if plan.is_upgrade:
            dlg = UpdateDialog(window, status, plan)
            window.update_dialog = dlg
            dlg.show()
        elif plan.fetch:
            offer_finish_setup(window, status)
        elif manual:
            _info(window, "Updates", f"You have the latest version of Cutout ({status.version}).")

    def failed(exc):
        window.statusBar().clearMessage()
        log.info("update check failed: %s", exc)
        if manual:
            _warn(window, "Couldn't check for updates", str(exc))

    window.run_in_background(client.fetch_manifest, on_done=got, on_error=failed)


def _info(window, title: str, text: str) -> None:
    if getattr(window, "ask_user", False):
        QMessageBox.information(window, title, text)
    else:
        window.statusBar().showMessage(text, 8000)


def _warn(window, title: str, text: str) -> None:
    if getattr(window, "ask_user", False):
        QMessageBox.warning(window, title, text)
    else:
        window.statusBar().showMessage(text, 8000)


# --------------------------------------------------------------- dialog


class DownloadThread(QThread):
    progressed = Signal(object, object, str)  # done bytes, total bytes, what (ints can pass 2 GB)
    succeeded = Signal(object)  # list of (src, dest)
    errored = Signal(str)

    def __init__(self, plan: client.Plan, mirrors=()):
        super().__init__()
        self.plan = plan
        self.mirrors = mirrors
        self._stop = threading.Event()

    def cancel(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            adds = client.Downloader(self.plan, client.work_dir(self.plan.target),
                                     progress=lambda d, t, w: self.progressed.emit(d, t, w),
                                     cancel=self._stop.is_set, mirrors=self.mirrors).run()
        except client.Cancelled:
            self.errored.emit("")
        except client.UpdateError as exc:
            self.errored.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - shown to the user, full trace in the log
            log.exception("update download failed")
            self.errored.emit(f"The update could not be downloaded ({exc}).")
        else:
            self.succeeded.emit(adds)


class UpdateDialog(QDialog):
    """Offer -> download with progress -> close the app and let the helper install."""

    def __init__(self, window, status: client.Status, plan: client.Plan, first_run: bool = False):
        super().__init__(window)
        self.window_ = window
        self.status = status
        self.plan = plan
        self.first_run = first_run
        self.thread: DownloadThread | None = None
        self.adds = None
        self.setWindowTitle("Finish setting up Cutout" if first_run else "Update available")
        self.setMinimumWidth(480)

        lay = QVBoxLayout(self)
        size = _size(plan.download_size)
        if first_run:
            heading = "Cutout needs to download its AI models and GPU support once."
            body = (f"That is {size}. It can take a while on a slow connection; you can keep this window "
                    "open and it continues where it left off if the connection drops. "
                    "After that, Cutout works fully offline.")
        else:
            heading = f"Cutout {plan.target.version} is available. You have {status.version}."
            body = f"The update is a {size} download. Cutout restarts to finish, and your settings are kept."
        self.heading = QLabel(f"<b>{heading}</b>")
        self.heading.setWordWrap(True)
        self.body = QLabel(body)
        self.body.setWordWrap(True)
        lay.addWidget(self.heading)
        lay.addWidget(self.body)

        self.notes = QTextBrowser()
        self.notes.setOpenExternalLinks(True)
        self.notes.setMarkdown(plan.target.notes or "")
        self.notes.setVisible(bool(plan.target.notes) and not first_run)
        self.notes.setMinimumHeight(160)
        lay.addWidget(self.notes)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setVisible(False)
        self.what = QLabel("")
        self.what.setVisible(False)
        lay.addWidget(self.what)
        lay.addWidget(self.bar)

        self.buttons = QDialogButtonBox()
        self.btn_go = QPushButton("Download" if first_run else "Install Update")
        self.btn_go.setObjectName("primaryButton")
        self.btn_go.setDefault(True)
        self.btn_later = QPushButton("Later")
        self.buttons.addButton(self.btn_go, QDialogButtonBox.ButtonRole.AcceptRole)
        self.buttons.addButton(self.btn_later, QDialogButtonBox.ButtonRole.RejectRole)
        self.btn_go.clicked.connect(lambda: self.start())
        self.btn_later.clicked.connect(self.reject)
        lay.addWidget(self.buttons)

    # -- steps

    def start(self, mirrors=()) -> None:
        if self.adds is not None:  # already downloaded, the window refused to close last time
            self.install()
            return
        if not client.writable(self.status.root):
            self._fail("Cutout is installed in a protected folder, so it cannot update itself. "
                       f"Download the latest Setup from {RELEASES_PAGE} instead.")
            return
        self.btn_go.setEnabled(False)
        self.btn_later.setText("Cancel")
        self.bar.setVisible(True)
        self.what.setVisible(True)
        self.what.setText("Starting download…")
        self.thread = DownloadThread(self.plan, mirrors)
        self.thread.progressed.connect(self._progress)
        self.thread.succeeded.connect(self._downloaded)
        self.thread.errored.connect(self._fail)
        self.thread.start()

    def _progress(self, done, total, what) -> None:
        self.bar.setValue(int(done * 1000 / total) if total else 1000)
        self.what.setText(f"{what}: {_size(done)} of {_size(total)}")

    def _downloaded(self, adds) -> None:
        self.adds = adds
        self.bar.setValue(1000)
        self.what.setText("Download complete. Cutout will now restart to finish.")
        self.btn_later.setText("Later")
        self.install()

    def install(self) -> None:
        win = self.window_
        plan_path = client.write_apply_plan(
            self.status.root, self.plan, self.adds, client.work_dir(self.plan.target),
            restart=self.status.root / "Cutout.exe", wait_pid=os.getpid())
        self.hide()
        if not win.close():  # unsaved picture and the user chose to keep it open
            self.show()
            self.btn_go.setEnabled(True)
            self.btn_go.setText("Restart and Install")
            self.what.setText("Save or close your picture, then click Restart and Install.")
            return
        try:
            client.launch_apply(plan_path)
        except (client.UpdateError, OSError) as exc:
            win.show()
            self.show()
            self._fail(f"The installer helper could not start ({exc}).")
            return
        QApplication.quit()

    def _fail(self, message: str) -> None:
        self.btn_go.setEnabled(True)
        self.btn_go.setText("Try Again")
        self.btn_later.setText("Later")
        self.bar.setVisible(False)
        self.what.setVisible(True)
        self.what.setText(message or "Download paused. Click Try Again to continue where it stopped.")
        self.thread = None

    def reject(self) -> None:
        if self.thread is not None and self.thread.isRunning():
            self.thread.cancel()
            return  # errored("") arrives once it has stopped, and offers Try Again
        super().reject()
