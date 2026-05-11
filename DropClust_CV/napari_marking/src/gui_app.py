from __future__ import annotations

import json
import os
import sys
import csv
from pathlib import Path
from typing import Optional, Union

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyQt5 is required for gui_app.py. Install it in the GUI environment, "
        "for example: python -m pip install PyQt5"
    ) from exc

try:
    from PyQt5 import QtMultimedia, QtMultimediaWidgets
    MULTIMEDIA_AVAILABLE = True
except ImportError:  # pragma: no cover
    QtMultimedia = None
    QtMultimediaWidgets = None
    MULTIMEDIA_AVAILABLE = False


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SINGLE_OUTPUT = PROJECT_DIR / "artifacts" / "gui_run"
DEFAULT_BATCH_VIDEOS = PROJECT_DIR / "mlops" / "input" / "videos"
DEFAULT_BATCH_ANNOTATIONS = PROJECT_DIR / "mlops" / "input" / "annotations"
DEFAULT_BATCH_OUTPUT = PROJECT_DIR / "mlops" / "output"
DEFAULT_VIDEO = PROJECT_DIR / "data" / "raw" / "input.avi"
DEFAULT_ANNOTATIONS = PROJECT_DIR / "data" / "annotations" / "manual_points.csv"
DEFAULT_PIPELINE_PYTHON = PROJECT_DIR / "venv" / "bin" / "python"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TEXT_SUFFIXES = {".json", ".csv", ".txt", ".log", ".md"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}

RUN_COLOR_GOOD = QtGui.QColor("#dff6dd")
RUN_COLOR_MEDIUM = QtGui.QColor("#fff3cd")
RUN_COLOR_BAD = QtGui.QColor("#f8d7da")
RUN_COLOR_FAILED = QtGui.QColor("#f1aeb5")


def default_pipeline_python() -> Path:
    env_override = os.environ.get("DROPLET_PIPELINE_PYTHON")
    if env_override:
        return Path(env_override).expanduser()
    if DEFAULT_PIPELINE_PYTHON.exists():
        return DEFAULT_PIPELINE_PYTHON
    return Path(sys.executable)


def open_in_file_manager(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))


def load_json(path: Path) -> Optional[Union[dict, list]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class PathRow(QtWidgets.QWidget):
    def __init__(
        self,
        label: str,
        initial_value: str,
        pick_mode: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.pick_mode = pick_mode

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QtWidgets.QLabel(label)
        title.setMinimumWidth(120)
        self.edit = QtWidgets.QLineEdit(initial_value)
        self.button = QtWidgets.QPushButton("Browse")
        self.button.clicked.connect(self.pick)

        layout.addWidget(title)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

    def text(self) -> str:
        return self.edit.text().strip()

    def set_enabled(self, enabled: bool) -> None:
        self.edit.setEnabled(enabled)
        self.button.setEnabled(enabled)

    def pick(self) -> None:
        if self.pick_mode == "file":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select file",
                str(PROJECT_DIR),
                "Files (*.avi *.mp4 *.mov *.mkv *.mpeg *.mpg *.csv);;All files (*)",
            )
        else:
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Select directory",
                str(PROJECT_DIR),
            )
        if path:
            self.edit.setText(path)


class DropletCvGui(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Droplet CV Workbench")
        self.resize(1120, 780)

        self.process: Optional[QtCore.QProcess] = None
        self.current_kind: Optional[str] = None

        self.pipeline_python_row = PathRow(
            "Pipeline Python",
            str(default_pipeline_python()),
            "file",
        )

        self.single_video_row = PathRow("Video", str(DEFAULT_VIDEO), "file")
        self.single_annotations_row = PathRow("Annotations", str(DEFAULT_ANNOTATIONS), "file")
        self.single_output_row = PathRow("Output", str(DEFAULT_SINGLE_OUTPUT), "dir")

        self.batch_videos_row = PathRow("Videos Dir", str(DEFAULT_BATCH_VIDEOS), "dir")
        self.batch_annotations_row = PathRow(
            "Annotations Dir",
            str(DEFAULT_BATCH_ANNOTATIONS),
            "dir",
        )
        self.batch_output_row = PathRow("Output Dir", str(DEFAULT_BATCH_OUTPUT), "dir")

        self.status_label = QtWidgets.QLabel("Ready")
        self.single_mode_combo = QtWidgets.QComboBox()
        self.single_mode_combo.addItems(["balanced", "high-recall", "baseline"])
        self.single_use_annotations = QtWidgets.QCheckBox("Use annotations and compute metrics")
        self.single_skip_tuning = QtWidgets.QCheckBox("Skip tuning and use current detector presets")
        self.single_skip_tuning.setChecked(True)
        self.single_use_annotations.stateChanged.connect(self._toggle_single_annotations)

        self.batch_mode_combo = QtWidgets.QComboBox()
        self.batch_mode_combo.addItems(["balanced", "high-recall", "baseline"])
        self.batch_overwrite = QtWidgets.QCheckBox("Overwrite existing output folders")
        self.batch_overwrite.setChecked(True)

        self.single_summary = QtWidgets.QPlainTextEdit()
        self.single_summary.setReadOnly(True)
        self.batch_summary = QtWidgets.QPlainTextEdit()
        self.batch_summary.setReadOnly(True)
        self.log_output = QtWidgets.QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.output_root_label = QtWidgets.QLabel("Output browser is empty.")
        self.output_tree = QtWidgets.QTreeWidget()
        self.output_tree.setHeaderLabels(["Artifact", "Type"])
        self.output_tree.itemSelectionChanged.connect(self._handle_output_selection)
        self.image_scroll = QtWidgets.QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_label = QtWidgets.QLabel("No image selected")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        self.image_scroll.setWidget(self.image_label)
        self.text_preview = QtWidgets.QPlainTextEdit()
        self.text_preview.setReadOnly(True)
        self.preview_stack = QtWidgets.QStackedWidget()
        self.current_preview_pixmap: Optional[QtGui.QPixmap] = None
        self.current_output_root: Optional[Path] = None
        self.current_preview_path: Optional[Path] = None
        self.qc_gallery_root_label = QtWidgets.QLabel("QC gallery is empty.")
        self.qc_list = QtWidgets.QListWidget()
        self.qc_list.currentRowChanged.connect(self._handle_qc_row_changed)
        self.qc_image_scroll = QtWidgets.QScrollArea()
        self.qc_image_scroll.setWidgetResizable(True)
        self.qc_image_label = QtWidgets.QLabel("No QC frame selected")
        self.qc_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.qc_image_label.setMinimumSize(320, 240)
        self.qc_image_scroll.setWidget(self.qc_image_label)
        self.qc_info = QtWidgets.QPlainTextEdit()
        self.qc_info.setReadOnly(True)
        self.current_qc_pixmap: Optional[QtGui.QPixmap] = None
        self.current_qc_files: list[Path] = []
        self.current_qc_path: Optional[Path] = None
        self.qc_problem_rows: dict[int, dict[str, str]] = {}
        self.run_history_table = QtWidgets.QTableWidget(0, 8)
        self.run_history_table.setHorizontalHeaderLabels(
            ["Kind", "Name", "Mode", "Status", "Precision", "Recall", "F1", "Output"]
        )
        self.run_history_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.run_history_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.run_history_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.run_history_table.setAlternatingRowColors(True)
        self.run_history_table.horizontalHeader().setStretchLastSection(True)
        self.run_history_info = QtWidgets.QPlainTextEdit()
        self.run_history_info.setReadOnly(True)
        self.run_history_rows: list[dict[str, object]] = []
        self.good_f1_spin = QtWidgets.QDoubleSpinBox()
        self.good_f1_spin.setRange(0.0, 1.0)
        self.good_f1_spin.setSingleStep(0.01)
        self.good_f1_spin.setDecimals(3)
        self.good_f1_spin.setValue(0.87)
        self.good_recall_spin = QtWidgets.QDoubleSpinBox()
        self.good_recall_spin.setRange(0.0, 1.0)
        self.good_recall_spin.setSingleStep(0.01)
        self.good_recall_spin.setDecimals(3)
        self.good_recall_spin.setValue(0.84)
        self.medium_f1_spin = QtWidgets.QDoubleSpinBox()
        self.medium_f1_spin.setRange(0.0, 1.0)
        self.medium_f1_spin.setSingleStep(0.01)
        self.medium_f1_spin.setDecimals(3)
        self.medium_f1_spin.setValue(0.84)
        self.medium_precision_spin = QtWidgets.QDoubleSpinBox()
        self.medium_precision_spin.setRange(0.0, 1.0)
        self.medium_precision_spin.setSingleStep(0.01)
        self.medium_precision_spin.setDecimals(3)
        self.medium_precision_spin.setValue(0.80)
        self.video_player = None
        self.video_widget = None
        self.video_message = QtWidgets.QLabel()
        self.video_message.setAlignment(QtCore.Qt.AlignCenter)
        if MULTIMEDIA_AVAILABLE:
            self.video_widget = QtMultimediaWidgets.QVideoWidget()
            self.video_player = QtMultimedia.QMediaPlayer(self)
            self.video_player.setVideoOutput(self.video_widget)

        self._build_ui()
        self._toggle_single_annotations()
        self.refresh_single_summary()
        self.refresh_batch_summary()
        self.refresh_run_history()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        header = QtWidgets.QHBoxLayout()
        title_box = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("Droplet CV Workbench")
        title_font = title.font()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        subtitle = QtWidgets.QLabel(
            "Single-video inference, batch jobs and quick access to recent outputs."
        )
        subtitle.setStyleSheet("color: #555;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box, 1)
        header.addWidget(self.status_label)
        root.addLayout(header)

        pipeline_group = QtWidgets.QGroupBox("Runtime")
        pipeline_layout = QtWidgets.QVBoxLayout(pipeline_group)
        pipeline_layout.addWidget(self.pipeline_python_row)
        runtime_hint = QtWidgets.QLabel(
            "GUI may run in one environment, while the actual CV pipeline can run in another."
        )
        runtime_hint.setStyleSheet("color: #666;")
        pipeline_layout.addWidget(runtime_hint)
        root.addWidget(pipeline_group)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_single_tab(), "Single Video")
        tabs.addTab(self._build_batch_tab(), "Batch")
        root.addWidget(tabs, 3)

        bottom_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        top_bottom = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        top_bottom.addWidget(self._build_summary_widget())
        top_bottom.addWidget(self._build_output_tools_widget())
        top_bottom.setSizes([420, 620])
        bottom_splitter.addWidget(top_bottom)
        bottom_splitter.addWidget(self._build_log_widget())
        bottom_splitter.setSizes([360, 220])
        root.addWidget(bottom_splitter, 2)

    def _build_single_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        group = QtWidgets.QGroupBox("Single Video Run")
        form = QtWidgets.QVBoxLayout(group)
        form.addWidget(self.single_video_row)
        form.addWidget(self.single_annotations_row)
        form.addWidget(self.single_output_row)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Mode"))
        mode_row.addWidget(self.single_mode_combo)
        mode_row.addStretch(1)
        form.addLayout(mode_row)

        form.addWidget(self.single_use_annotations)
        form.addWidget(self.single_skip_tuning)

        actions = QtWidgets.QHBoxLayout()
        run_button = QtWidgets.QPushButton("Run Single")
        run_button.clicked.connect(self.run_single)
        open_button = QtWidgets.QPushButton("Open Output Folder")
        open_button.clicked.connect(self.open_single_output)
        refresh_button = QtWidgets.QPushButton("Load Summary")
        refresh_button.clicked.connect(self.refresh_single_summary)
        actions.addWidget(run_button)
        actions.addWidget(open_button)
        actions.addWidget(refresh_button)
        actions.addStretch(1)
        form.addLayout(actions)

        layout.addWidget(group)
        layout.addStretch(1)
        return widget

    def _build_batch_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        group = QtWidgets.QGroupBox("Batch Run")
        form = QtWidgets.QVBoxLayout(group)
        form.addWidget(self.batch_videos_row)
        form.addWidget(self.batch_annotations_row)
        form.addWidget(self.batch_output_row)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Mode"))
        mode_row.addWidget(self.batch_mode_combo)
        mode_row.addStretch(1)
        form.addLayout(mode_row)

        form.addWidget(self.batch_overwrite)

        actions = QtWidgets.QHBoxLayout()
        run_button = QtWidgets.QPushButton("Run Batch")
        run_button.clicked.connect(self.run_batch)
        open_button = QtWidgets.QPushButton("Open Output Folder")
        open_button.clicked.connect(self.open_batch_output)
        refresh_button = QtWidgets.QPushButton("Load Manifest")
        refresh_button.clicked.connect(self.refresh_batch_summary)
        actions.addWidget(run_button)
        actions.addWidget(open_button)
        actions.addWidget(refresh_button)
        actions.addStretch(1)
        form.addLayout(actions)

        layout.addWidget(group)
        layout.addStretch(1)
        return widget

    def _build_summary_widget(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)

        single_box = QtWidgets.QGroupBox("Latest Single Summary")
        single_layout = QtWidgets.QVBoxLayout(single_box)
        single_layout.addWidget(self.single_summary)

        batch_box = QtWidgets.QGroupBox("Latest Batch Summary")
        batch_layout = QtWidgets.QVBoxLayout(batch_box)
        batch_layout.addWidget(self.batch_summary)

        layout.addWidget(single_box)
        layout.addWidget(batch_box)
        return widget

    def _build_log_widget(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        box = QtWidgets.QGroupBox("Run Log")
        box_layout = QtWidgets.QVBoxLayout(box)
        box_layout.addWidget(self.log_output)
        layout.addWidget(box)
        return widget

    def _build_output_tools_widget(self) -> QtWidgets.QWidget:
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_output_browser_widget(), "Output Browser")
        tabs.addTab(self._build_qc_gallery_widget(), "QC Gallery")
        tabs.addTab(self._build_run_history_widget(), "Run History")
        return tabs

    def _build_output_browser_widget(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        box = QtWidgets.QGroupBox("Output Browser")
        box_layout = QtWidgets.QVBoxLayout(box)

        button_row = QtWidgets.QHBoxLayout()
        load_single_button = QtWidgets.QPushButton("Load Single Output")
        load_single_button.clicked.connect(lambda: self.populate_output_browser(
            Path(self.single_output_row.text()).expanduser()
        ))
        load_batch_button = QtWidgets.QPushButton("Load Batch Output")
        load_batch_button.clicked.connect(self._load_batch_browser_root)
        open_selected_button = QtWidgets.QPushButton("Open Selected")
        open_selected_button.clicked.connect(self.open_selected_artifact)
        button_row.addWidget(load_single_button)
        button_row.addWidget(load_batch_button)
        button_row.addWidget(open_selected_button)
        button_row.addStretch(1)

        self.output_root_label.setStyleSheet("color: #555;")
        box_layout.addLayout(button_row)
        box_layout.addWidget(self.output_root_label)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self.output_tree)

        preview_container = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(6)
        preview_layout.addWidget(self.preview_stack)

        self.preview_stack.addWidget(self.text_preview)
        self.preview_stack.addWidget(self.image_scroll)
        if self.video_widget is not None:
            self.preview_stack.addWidget(self.video_widget)
        else:
            self.video_message.setText("Video preview is unavailable in this Qt build.")
            self.preview_stack.addWidget(self.video_message)

        split.addWidget(preview_container)
        split.setSizes([260, 420])
        box_layout.addWidget(split)
        layout.addWidget(box)
        return widget

    def _build_qc_gallery_widget(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        box = QtWidgets.QGroupBox("QC Gallery")
        box_layout = QtWidgets.QVBoxLayout(box)

        button_row = QtWidgets.QHBoxLayout()
        load_single_button = QtWidgets.QPushButton("Load Single QC")
        load_single_button.clicked.connect(
            lambda: self.populate_qc_gallery(Path(self.single_output_row.text()).expanduser())
        )
        load_batch_button = QtWidgets.QPushButton("Load Batch QC")
        load_batch_button.clicked.connect(self._load_batch_qc_root)
        prev_button = QtWidgets.QPushButton("Prev")
        prev_button.clicked.connect(self._select_prev_qc)
        next_button = QtWidgets.QPushButton("Next")
        next_button.clicked.connect(self._select_next_qc)
        open_button = QtWidgets.QPushButton("Open Current")
        open_button.clicked.connect(self.open_current_qc)
        button_row.addWidget(load_single_button)
        button_row.addWidget(load_batch_button)
        button_row.addWidget(prev_button)
        button_row.addWidget(next_button)
        button_row.addWidget(open_button)
        button_row.addStretch(1)

        self.qc_gallery_root_label.setStyleSheet("color: #555;")
        box_layout.addLayout(button_row)
        box_layout.addWidget(self.qc_gallery_root_label)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self.qc_list)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        right_layout.addWidget(self.qc_image_scroll, 3)
        right_layout.addWidget(self.qc_info, 1)
        split.addWidget(right)
        split.setSizes([220, 460])

        box_layout.addWidget(split)
        layout.addWidget(box)
        return widget

    def _build_run_history_widget(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        box = QtWidgets.QGroupBox("Run History")
        box_layout = QtWidgets.QVBoxLayout(box)

        thresholds_row = QtWidgets.QHBoxLayout()
        thresholds_row.addWidget(QtWidgets.QLabel("Good F1"))
        thresholds_row.addWidget(self.good_f1_spin)
        thresholds_row.addWidget(QtWidgets.QLabel("Good Recall"))
        thresholds_row.addWidget(self.good_recall_spin)
        thresholds_row.addWidget(QtWidgets.QLabel("Medium F1"))
        thresholds_row.addWidget(self.medium_f1_spin)
        thresholds_row.addWidget(QtWidgets.QLabel("Medium Precision"))
        thresholds_row.addWidget(self.medium_precision_spin)
        thresholds_row.addStretch(1)

        button_row = QtWidgets.QHBoxLayout()
        refresh_button = QtWidgets.QPushButton("Refresh History")
        refresh_button.clicked.connect(self.refresh_run_history)
        reopen_button = QtWidgets.QPushButton("Reopen Selected Run")
        reopen_button.clicked.connect(self.reopen_selected_run)
        load_browser_button = QtWidgets.QPushButton("Load Browser")
        load_browser_button.clicked.connect(self.load_selected_run_browser)
        load_qc_button = QtWidgets.QPushButton("Load QC")
        load_qc_button.clicked.connect(self.load_selected_run_qc)
        open_output_button = QtWidgets.QPushButton("Open Output Folder")
        open_output_button.clicked.connect(self.open_selected_run_output)
        button_row.addWidget(refresh_button)
        button_row.addWidget(reopen_button)
        button_row.addWidget(load_browser_button)
        button_row.addWidget(load_qc_button)
        button_row.addWidget(open_output_button)
        button_row.addStretch(1)

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.addWidget(self.run_history_table)
        split.addWidget(self.run_history_info)
        split.setSizes([260, 120])

        legend = QtWidgets.QLabel(
            "Green = good, yellow = medium, red = weak, dark red = failed."
        )
        legend.setStyleSheet("color: #666;")

        box_layout.addLayout(thresholds_row)
        box_layout.addLayout(button_row)
        box_layout.addWidget(legend)
        box_layout.addWidget(split)
        layout.addWidget(box)

        self.run_history_table.itemSelectionChanged.connect(self._handle_run_history_selection)
        self.good_f1_spin.valueChanged.connect(self.refresh_run_history)
        self.good_recall_spin.valueChanged.connect(self.refresh_run_history)
        self.medium_f1_spin.valueChanged.connect(self.refresh_run_history)
        self.medium_precision_spin.valueChanged.connect(self.refresh_run_history)
        return widget

    def _toggle_single_annotations(self) -> None:
        enabled = self.single_use_annotations.isChecked()
        self.single_annotations_row.set_enabled(enabled)

    def _append_log(self, text: str) -> None:
        self.log_output.appendPlainText(text.rstrip())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_image_preview()
        self._update_qc_image_preview()

    def _pipeline_python(self) -> Path:
        return Path(self.pipeline_python_row.text()).expanduser()

    def _ensure_pipeline_python(self) -> Optional[Path]:
        python_path = self._pipeline_python()
        if not python_path.exists():
            QtWidgets.QMessageBox.critical(
                self,
                "Missing interpreter",
                "Pipeline Python not found:\n%s" % python_path,
            )
            return None
        return python_path

    def _start_process(self, command: list[str], kind: str) -> None:
        if self.process is not None:
            QtWidgets.QMessageBox.information(self, "Busy", "Another run is already in progress.")
            return

        self.current_kind = kind
        self.status_label.setText("Running %s..." % kind)
        self._append_log("$ " + " ".join(command))

        process = QtCore.QProcess(self)
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.setWorkingDirectory(str(PROJECT_DIR))
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._handle_process_output)
        process.finished.connect(self._handle_process_finished)
        self.process = process
        process.start()

    def _handle_process_output(self) -> None:
        if self.process is None:
            return
        data = self.process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        if data:
            self._append_log(data)

    def _handle_process_finished(self, exit_code: int, _status: QtCore.QProcess.ExitStatus) -> None:
        kind = self.current_kind or "run"
        if exit_code == 0:
            self.status_label.setText("%s completed" % kind.capitalize())
            if kind == "single":
                self.refresh_single_summary()
            elif kind == "batch":
                self.refresh_batch_summary()
        else:
            self.status_label.setText("%s failed" % kind.capitalize())
            QtWidgets.QMessageBox.critical(
                self,
                "Run failed",
                "%s exited with code %s.\nSee the log panel for details." % (kind, exit_code),
            )
        self.process = None
        self.current_kind = None
        self.refresh_run_history()

    def _stop_video_preview(self) -> None:
        if self.video_player is not None:
            self.video_player.stop()

    def _clear_preview(self, message: str) -> None:
        self._stop_video_preview()
        self.current_preview_path = None
        self.current_preview_pixmap = None
        self.text_preview.setPlainText(message)
        self.preview_stack.setCurrentWidget(self.text_preview)

    def _update_image_preview(self) -> None:
        if self.current_preview_pixmap is None:
            return
        scaled = self.current_preview_pixmap.scaled(
            self.image_scroll.viewport().size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def _update_qc_image_preview(self) -> None:
        if self.current_qc_pixmap is None:
            return
        scaled = self.current_qc_pixmap.scaled(
            self.qc_image_scroll.viewport().size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.qc_image_label.setPixmap(scaled)

    def _read_text_preview(self, path: Path, max_chars: int = 30000) -> str:
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            return text[:max_chars] + "\n\n... truncated ..."
        return text

    def _preferred_preview_file(self, root_dir: Path) -> Optional[Path]:
        candidates = sorted(root_dir.rglob("*"))
        priority_names = [
            "qc_frame_rank_01",
            "preview_frame_0",
            "run_summary.json",
            "manifest.json",
            "qc_overlay.mp4",
        ]
        for marker in priority_names:
            for path in candidates:
                if path.is_file() and marker in path.name:
                    return path
        for path in candidates:
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                return path
        for path in candidates:
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                return path
        for path in candidates:
            if path.is_file():
                return path
        return None

    def _load_qc_problem_rows(self, root_dir: Path) -> dict[int, dict[str, str]]:
        candidates = [
            root_dir / "tables" / "qc_problem_frames.csv",
            root_dir / "qc_problem_frames.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                with candidate.open("r", encoding="utf-8", errors="replace") as handle:
                    reader = csv.DictReader(handle)
                    rows: dict[int, dict[str, str]] = {}
                    for row in reader:
                        frame_text = row.get("frame")
                        if frame_text is None:
                            continue
                        try:
                            rows[int(frame_text)] = row
                        except ValueError:
                            continue
                    return rows
        return {}

    def populate_output_browser(self, root_dir: Path) -> None:
        root_dir = root_dir.expanduser()
        self.output_tree.clear()
        self.current_output_root = root_dir
        self.output_root_label.setText("Current root: %s" % root_dir)
        if not root_dir.exists():
            self._clear_preview("Output directory does not exist:\n%s" % root_dir)
            return

        def add_children(parent_item: QtWidgets.QTreeWidgetItem, directory: Path) -> None:
            for child in sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
                item = QtWidgets.QTreeWidgetItem(
                    [child.name, "file" if child.is_file() else "dir"]
                )
                item.setData(0, QtCore.Qt.UserRole, str(child))
                parent_item.addChild(item)
                if child.is_dir():
                    add_children(item, child)

        root_item = QtWidgets.QTreeWidgetItem([root_dir.name, "dir"])
        root_item.setData(0, QtCore.Qt.UserRole, str(root_dir))
        self.output_tree.addTopLevelItem(root_item)
        add_children(root_item, root_dir)
        root_item.setExpanded(True)

        preferred = self._preferred_preview_file(root_dir)
        if preferred is not None:
            matches = self.output_tree.findItems(
                preferred.name,
                QtCore.Qt.MatchRecursive | QtCore.Qt.MatchExactly,
                0,
            )
            for item in matches:
                if item.data(0, QtCore.Qt.UserRole) == str(preferred):
                    self.output_tree.setCurrentItem(item)
                    return
        self._clear_preview("No previewable artifacts found in:\n%s" % root_dir)

    def populate_qc_gallery(self, root_dir: Path) -> None:
        root_dir = root_dir.expanduser()
        qc_dir = root_dir / "qc"
        self.current_qc_files = []
        self.current_qc_path = None
        self.current_qc_pixmap = None
        self.qc_list.clear()
        self.qc_image_label.setText("No QC frame selected")
        self.qc_image_label.setPixmap(QtGui.QPixmap())
        self.qc_problem_rows = self._load_qc_problem_rows(root_dir)
        self.qc_gallery_root_label.setText("Current QC root: %s" % qc_dir)

        if not qc_dir.exists():
            self.qc_info.setPlainText("QC directory does not exist:\n%s" % qc_dir)
            return

        preferred_files = sorted(qc_dir.glob("qc_frame_rank_*.png"))
        if not preferred_files:
            preview = qc_dir / "preview_frame_0.png"
            if preview.exists():
                preferred_files = [preview]
        if not preferred_files:
            preferred_files = sorted([p for p in qc_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES])

        self.current_qc_files = preferred_files
        for path in preferred_files:
            item_text = path.name
            frame_info = self._frame_from_qc_name(path)
            if frame_info is not None and frame_info in self.qc_problem_rows:
                row = self.qc_problem_rows[frame_info]
                item_text = "%s | TP %s FP %s FN %s" % (
                    path.name,
                    row.get("tp", "?"),
                    row.get("fp", "?"),
                    row.get("fn", "?"),
                )
            item = QtWidgets.QListWidgetItem(item_text)
            item.setData(QtCore.Qt.UserRole, str(path))
            self.qc_list.addItem(item)

        if preferred_files:
            self.qc_list.setCurrentRow(0)
        else:
            self.qc_info.setPlainText("No QC PNG files found in:\n%s" % qc_dir)

    def _frame_from_qc_name(self, path: Path) -> Optional[int]:
        name = path.stem
        marker = "_frame_"
        if marker not in name:
            return None
        tail = name.split(marker, 1)[1]
        try:
            return int(tail)
        except ValueError:
            return None

    def _handle_qc_row_changed(self, index: int) -> None:
        if index < 0 or index >= len(self.current_qc_files):
            return
        self.preview_qc_path(self.current_qc_files[index])

    def preview_qc_path(self, path: Path) -> None:
        pixmap = QtGui.QPixmap(str(path))
        self.current_qc_path = path
        if pixmap.isNull():
            self.current_qc_pixmap = None
            self.qc_image_label.setText("Could not render image:\n%s" % path)
            self.qc_info.setPlainText(str(path))
            return
        self.current_qc_pixmap = pixmap
        self._update_qc_image_preview()
        frame_id = self._frame_from_qc_name(path)
        info_lines = [str(path)]
        if frame_id is not None:
            info_lines.append("Frame: %s" % frame_id)
            if frame_id in self.qc_problem_rows:
                row = self.qc_problem_rows[frame_id]
                ordered = ["rank", "frame", "tp", "fp", "fn", "precision", "recall", "f1"]
                for key in ordered:
                    if key in row and row[key] != "":
                        info_lines.append("%s: %s" % (key, row[key]))
        self.qc_info.setPlainText("\n".join(info_lines))

    def _select_prev_qc(self) -> None:
        row = self.qc_list.currentRow()
        if row > 0:
            self.qc_list.setCurrentRow(row - 1)

    def _select_next_qc(self) -> None:
        row = self.qc_list.currentRow()
        if 0 <= row < self.qc_list.count() - 1:
            self.qc_list.setCurrentRow(row + 1)

    def open_current_qc(self) -> None:
        if self.current_qc_path is not None:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.current_qc_path)))

    def _handle_output_selection(self) -> None:
        items = self.output_tree.selectedItems()
        if not items:
            return
        path = Path(items[0].data(0, QtCore.Qt.UserRole))
        if path.is_dir():
            self._clear_preview("Directory selected:\n%s" % path)
            return
        self.preview_artifact(path)

    def preview_artifact(self, path: Path) -> None:
        suffix = path.suffix.lower()
        self.current_preview_path = path
        if suffix in IMAGE_SUFFIXES:
            pixmap = QtGui.QPixmap(str(path))
            if pixmap.isNull():
                self._clear_preview("Could not render image:\n%s" % path)
                return
            self._stop_video_preview()
            self.current_preview_pixmap = pixmap
            self.preview_stack.setCurrentWidget(self.image_scroll)
            self._update_image_preview()
            return

        if suffix in TEXT_SUFFIXES:
            self._stop_video_preview()
            self.current_preview_pixmap = None
            self.text_preview.setPlainText(self._read_text_preview(path))
            self.preview_stack.setCurrentWidget(self.text_preview)
            return

        if suffix in VIDEO_SUFFIXES:
            self.current_preview_pixmap = None
            if self.video_player is not None and self.video_widget is not None:
                media = QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(str(path)))
                self.video_player.setMedia(media)
                self.preview_stack.setCurrentWidget(self.video_widget)
                self.video_player.play()
            else:
                self.video_message.setText(
                    "Video preview is unavailable in this Qt build.\n\n%s" % path
                )
                self.preview_stack.setCurrentWidget(self.video_message)
            return

        self._stop_video_preview()
        self.current_preview_pixmap = None
        self.text_preview.setPlainText(
            "Preview is not implemented for this file type.\n\n%s" % path
        )
        self.preview_stack.setCurrentWidget(self.text_preview)

    def open_selected_artifact(self) -> None:
        items = self.output_tree.selectedItems()
        if not items:
            return
        path = Path(items[0].data(0, QtCore.Qt.UserRole))
        if path.is_dir():
            open_in_file_manager(path)
        else:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def _load_batch_browser_root(self) -> None:
        output_dir = Path(self.batch_output_row.text()).expanduser()
        manifest_path = output_dir / "manifest.json"
        manifest = load_json(manifest_path)
        if isinstance(manifest, list) and manifest:
            latest_dir = manifest[-1].get("output_dir")
            if latest_dir:
                self.populate_output_browser(Path(latest_dir))
                return
        self.populate_output_browser(output_dir)

    def _load_batch_qc_root(self) -> None:
        output_dir = Path(self.batch_output_row.text()).expanduser()
        manifest_path = output_dir / "manifest.json"
        manifest = load_json(manifest_path)
        if isinstance(manifest, list) and manifest:
            latest_dir = manifest[-1].get("output_dir")
            if latest_dir:
                self.populate_qc_gallery(Path(latest_dir))
                return
        self.populate_qc_gallery(output_dir)

    def _format_metric(self, value: object) -> str:
        if value is None:
            return "-"
        try:
            return "%.3f" % float(value)
        except (TypeError, ValueError):
            return str(value)

    def _run_row_color(self, row: dict[str, object]) -> QtGui.QColor:
        status = str(row.get("status", "")).lower()
        if status == "failed":
            return RUN_COLOR_FAILED

        precision = row.get("precision")
        recall = row.get("recall")
        f1 = row.get("f1")
        try:
            precision_f = None if precision is None else float(precision)
            recall_f = None if recall is None else float(recall)
            f1_f = None if f1 is None else float(f1)
        except (TypeError, ValueError):
            return RUN_COLOR_MEDIUM

        if f1_f is None:
            return RUN_COLOR_MEDIUM
        if f1_f >= self.good_f1_spin.value() and (
            recall_f is None or recall_f >= self.good_recall_spin.value()
        ):
            return RUN_COLOR_GOOD
        if f1_f >= self.medium_f1_spin.value() and (
            precision_f is None or precision_f >= self.medium_precision_spin.value()
        ):
            return RUN_COLOR_MEDIUM
        return RUN_COLOR_BAD

    def _discover_single_run_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        seen: set[Path] = set()
        candidate_roots = [PROJECT_DIR / "artifacts", Path(self.single_output_row.text()).expanduser()]
        for root in candidate_roots:
            if not root.exists():
                continue
            summary_paths = []
            direct_summary = root / "reports" / "run_summary.json"
            if direct_summary.exists():
                summary_paths.append(direct_summary)
            summary_paths.extend(root.glob("*/reports/run_summary.json"))
            for summary_path in summary_paths:
                resolved = summary_path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                summary = load_json(summary_path)
                if not isinstance(summary, dict):
                    continue
                metrics = summary.get("evaluation") or {}
                output_dir = summary_path.parent.parent
                rows.append(
                    {
                        "kind": "single",
                        "name": output_dir.name,
                        "mode": summary.get("mode"),
                        "status": "ok",
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1": metrics.get("f1"),
                        "output_dir": str(output_dir),
                        "video": summary.get("video", ""),
                        "annotation": "",
                    }
                )
        return rows

    def _discover_batch_run_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        seen: set[Path] = set()
        manifest_paths = [
            Path(self.batch_output_row.text()).expanduser() / "manifest.json",
            PROJECT_DIR / "mlops" / "output" / "manifest.json",
        ]
        for manifest_path in manifest_paths:
            resolved = manifest_path.resolve()
            if resolved in seen or not manifest_path.exists():
                continue
            seen.add(resolved)
            manifest = load_json(manifest_path)
            if not isinstance(manifest, list):
                continue
            for entry in manifest:
                if not isinstance(entry, dict):
                    continue
                rows.append(
                    {
                        "kind": "batch",
                        "name": entry.get("job_name", ""),
                        "mode": entry.get("mode", ""),
                        "status": entry.get("status", ""),
                        "precision": entry.get("precision"),
                        "recall": entry.get("recall"),
                        "f1": entry.get("f1"),
                        "output_dir": entry.get("output_dir", ""),
                        "video": entry.get("video", ""),
                        "annotation": entry.get("annotation", ""),
                    }
                )
        return rows

    def refresh_run_history(self) -> None:
        rows = self._discover_single_run_rows() + self._discover_batch_run_rows()
        rows.sort(key=lambda row: str(row.get("output_dir", "")), reverse=True)
        self.run_history_rows = rows
        self.run_history_table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            background = self._run_row_color(row)
            values = [
                str(row.get("kind", "")),
                str(row.get("name", "")),
                str(row.get("mode", "")),
                str(row.get("status", "")),
                self._format_metric(row.get("precision")),
                self._format_metric(row.get("recall")),
                self._format_metric(row.get("f1")),
                str(row.get("output_dir", "")),
            ]
            for col_idx, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setData(QtCore.Qt.UserRole, row_idx)
                if col_idx in (4, 5, 6):
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                item.setBackground(background)
                self.run_history_table.setItem(row_idx, col_idx, item)
        if rows:
            self.run_history_table.selectRow(0)
        else:
            self.run_history_info.setPlainText("No runs discovered yet.")

    def _selected_run_row(self) -> Optional[dict[str, object]]:
        selected = self.run_history_table.selectedItems()
        if not selected:
            return None
        row_idx = selected[0].data(QtCore.Qt.UserRole)
        if row_idx is None:
            return None
        try:
            return self.run_history_rows[int(row_idx)]
        except (IndexError, ValueError, TypeError):
            return None

    def _handle_run_history_selection(self) -> None:
        row = self._selected_run_row()
        if row is None:
            return
        lines = [
            "Kind: %s" % row.get("kind", ""),
            "Name: %s" % row.get("name", ""),
            "Mode: %s" % row.get("mode", ""),
            "Status: %s" % row.get("status", ""),
            "Precision: %s" % self._format_metric(row.get("precision")),
            "Recall: %s" % self._format_metric(row.get("recall")),
            "F1: %s" % self._format_metric(row.get("f1")),
            "Output: %s" % row.get("output_dir", ""),
        ]
        if row.get("video"):
            lines.append("Video: %s" % row.get("video"))
        if row.get("annotation"):
            lines.append("Annotation: %s" % row.get("annotation"))
        self.run_history_info.setPlainText("\n".join(lines))

    def reopen_selected_run(self) -> None:
        row = self._selected_run_row()
        if row is None:
            return
        output_dir = str(row.get("output_dir", ""))
        if row.get("kind") == "batch":
            self.batch_output_row.edit.setText(str(Path(output_dir).parent))
            self.batch_mode_combo.setCurrentText(str(row.get("mode", "balanced")))
            self.refresh_batch_summary()
        else:
            self.single_output_row.edit.setText(output_dir)
            if row.get("video"):
                self.single_video_row.edit.setText(str(row.get("video")))
            self.single_mode_combo.setCurrentText(str(row.get("mode", "balanced")))
            self.refresh_single_summary()

    def load_selected_run_browser(self) -> None:
        row = self._selected_run_row()
        if row is None:
            return
        self.populate_output_browser(Path(str(row.get("output_dir", ""))))

    def load_selected_run_qc(self) -> None:
        row = self._selected_run_row()
        if row is None:
            return
        self.populate_qc_gallery(Path(str(row.get("output_dir", ""))))

    def open_selected_run_output(self) -> None:
        row = self._selected_run_row()
        if row is None:
            return
        open_in_file_manager(Path(str(row.get("output_dir", ""))))

    def run_single(self) -> None:
        python_path = self._ensure_pipeline_python()
        if python_path is None:
            return

        video = Path(self.single_video_row.text()).expanduser()
        output_dir = Path(self.single_output_row.text()).expanduser()
        if not video.exists():
            QtWidgets.QMessageBox.critical(self, "Missing video", "Video not found:\n%s" % video)
            return

        command = [
            str(python_path),
            str(PROJECT_DIR / "src" / "droplet_cv.py"),
            "--video",
            str(video),
            "--output-dir",
            str(output_dir),
            "--mode",
            self.single_mode_combo.currentText(),
        ]
        if self.single_skip_tuning.isChecked():
            command.append("--skip-tuning")

        if self.single_use_annotations.isChecked():
            annotations = Path(self.single_annotations_row.text()).expanduser()
            if not annotations.exists():
                QtWidgets.QMessageBox.critical(
                    self,
                    "Missing annotations",
                    "Annotation file not found:\n%s" % annotations,
                )
                return
            command.extend(["--annotations", str(annotations)])
        else:
            command.append("--no-annotations")

        self._start_process(command, "single")

    def run_batch(self) -> None:
        python_path = self._ensure_pipeline_python()
        if python_path is None:
            return

        videos_dir = Path(self.batch_videos_row.text()).expanduser()
        annotations_dir = Path(self.batch_annotations_row.text()).expanduser()
        output_dir = Path(self.batch_output_row.text()).expanduser()
        videos_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            str(python_path),
            str(PROJECT_DIR / "src" / "mlops_batch.py"),
            "--input-videos",
            str(videos_dir),
            "--input-annotations",
            str(annotations_dir),
            "--output-dir",
            str(output_dir),
            "--mode",
            self.batch_mode_combo.currentText(),
        ]
        if self.batch_overwrite.isChecked():
            command.append("--overwrite")

        self._start_process(command, "batch")

    def refresh_single_summary(self) -> None:
        output_dir = Path(self.single_output_row.text()).expanduser()
        summary_path = output_dir / "reports" / "run_summary.json"
        summary = load_json(summary_path)
        if summary is None:
            self.single_summary.setPlainText(
                "No run summary yet.\nExpected file:\n%s" % summary_path
            )
            return

        assert isinstance(summary, dict)
        metrics = summary.get("evaluation") or {}
        lines = [
            "Mode: %s" % summary.get("mode"),
            "Frames: %s" % summary.get("frame_count"),
            "Detections: %s" % summary.get("detections_total"),
            "Tracks: %s" % summary.get("tracks_total"),
            "Output: %s" % output_dir,
        ]
        if metrics:
            lines.extend(
                [
                    "Precision: %.3f" % metrics.get("precision"),
                    "Recall: %.3f" % metrics.get("recall"),
                    "F1: %.3f" % metrics.get("f1"),
                    "Count MAE: %.2f" % metrics.get("count_mae"),
                ]
            )
        else:
            lines.append("Metrics: not computed (no annotations)")
        lines.extend(
            [
                "Tables: %s" % (output_dir / "tables"),
                "QC: %s" % (output_dir / "qc"),
                "Report: %s" % summary_path,
            ]
        )
        self.single_summary.setPlainText("\n".join(lines))
        self.populate_output_browser(output_dir)
        self.populate_qc_gallery(output_dir)

    def refresh_batch_summary(self) -> None:
        output_dir = Path(self.batch_output_row.text()).expanduser()
        manifest_path = output_dir / "manifest.json"
        manifest = load_json(manifest_path)
        if manifest is None:
            self.batch_summary.setPlainText(
                "No manifest yet.\nExpected file:\n%s" % manifest_path
            )
            return

        assert isinstance(manifest, list)
        if not manifest:
            self.batch_summary.setPlainText("Manifest is empty:\n%s" % manifest_path)
            return

        ok_jobs = sum(1 for row in manifest if row.get("status") == "ok")
        failed_jobs = sum(1 for row in manifest if row.get("status") == "failed")
        latest = manifest[-1]
        lines = [
            "Jobs total: %s" % len(manifest),
            "OK: %s" % ok_jobs,
            "Failed: %s" % failed_jobs,
            "Latest job: %s" % latest.get("job_name"),
            "Mode: %s" % latest.get("mode"),
            "Output dir: %s" % latest.get("output_dir"),
        ]
        if latest.get("precision") is not None:
            lines.extend(
                [
                    "Precision: %.3f" % latest.get("precision"),
                    "Recall: %.3f" % latest.get("recall"),
                    "F1: %.3f" % latest.get("f1"),
                ]
            )
        else:
            lines.append("Metrics: not computed (no matching annotations)")
        lines.extend(
            [
                "Detections: %s" % latest.get("detections_total"),
                "Tracks: %s" % latest.get("tracks_total"),
                "Manifest: %s" % manifest_path,
            ]
        )
        self.batch_summary.setPlainText("\n".join(lines))
        latest_output = latest.get("output_dir")
        if latest_output:
            self.populate_output_browser(Path(latest_output))
            self.populate_qc_gallery(Path(latest_output))
        else:
            self.populate_output_browser(output_dir)
            self.populate_qc_gallery(output_dir)

    def open_single_output(self) -> None:
        open_in_file_manager(Path(self.single_output_row.text()).expanduser())

    def open_batch_output(self) -> None:
        open_in_file_manager(Path(self.batch_output_row.text()).expanduser())


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DropletCvGui()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
