from email.utils import getaddresses
from importlib.metadata import PackageMetadata, PackageNotFoundError, metadata, version

import numpy as np
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QDialog, QFormLayout, QLabel, QLineEdit

from miniml.gui.dialogs.common import finalize_dialog_window
from miniml.resources.util import get_app_icon_file_path


def get_package_metadata(dist_name: str = "miniml") -> dict:
    """
    Retrieve package metadata for GUI display with robust fallbacks.
    """
    fallback = {
        "version": "unknown",
        "authors": [],
        "urls": {},
    }

    try:
        package_meta: PackageMetadata = metadata(dist_name)
        package_version = version(dist_name)
    except PackageNotFoundError:
        return fallback

    authors = []

    for author in package_meta.get_all("Author", []) or []:
        cleaned_author = author.strip()
        if cleaned_author and cleaned_author not in authors:
            authors.append(cleaned_author)

    for author_email in package_meta.get_all("Author-email", []) or []:
        for name, email in getaddresses([author_email]):
            candidate = name.strip() if name and name.strip() else email.strip()
            if candidate and candidate not in authors:
                authors.append(candidate)

    urls = {}
    for project_url in package_meta.get_all("Project-URL", []) or []:
        if "," not in project_url:
            continue
        label, url = project_url.split(",", 1)
        label = label.strip()
        url = url.strip()
        if label and url:
            urls[label] = url

    return {
        "version": package_version,
        "authors": authors,
        "urls": urls,
    }


class FileInfoPanel(QDialog):
    """
    Read-only dialog that displays metadata for the loaded trace.
    """

    def __init__(
        self,
        *,
        trace_filename: str,
        filetype: str,
        total_time: float,
        y_unit: str,
        recording_mode: str,
        sampling_rate: float,
        protocol: str,
        parent=None,
    ):
        """
        Build and populate file information fields.
        """
        super().__init__(parent)

        self.filename = QLineEdit(trace_filename)
        self.filename.setReadOnly(True)
        self.filename.setFixedWidth(300)
        self.format = QLineEdit(filetype)
        self.format.setReadOnly(True)
        self.length = QLineEdit(f"{total_time:.2f}")
        self.length.setReadOnly(True)
        self.unit = QLineEdit(y_unit)
        self.unit.setReadOnly(True)
        self.mode = QLineEdit(recording_mode)
        self.mode.setReadOnly(True)
        self.sampling = QLineEdit(str(np.round(sampling_rate)))
        self.sampling.setReadOnly(True)
        self.protocol = QLineEdit(protocol)
        self.protocol.setReadOnly(True)
        self.protocol.setFixedWidth(300)

        layout = QFormLayout(self)
        layout.addRow("Filename:", self.filename)
        layout.addRow("File format:", self.format)
        layout.addRow("Recording duration (s):", self.length)
        layout.addRow("Data unit", self.unit)
        layout.addRow("Recording mode:", self.mode)
        layout.addRow("Sampling rate (Hz):", self.sampling)
        layout.addRow("Protocol:", self.protocol)
        self.setLayout(layout)

        finalize_dialog_window(self, title="File info", cancel=False)


class AboutPanel(QDialog):
    """
    About dialog with project metadata and external links.
    """

    def __init__(self, parent=None):
        """
        Build and populate the about panel layout.
        """
        super().__init__(parent)
        package_meta = get_package_metadata("miniml")

        layout = QFormLayout(self)

        logo_file_path = get_app_icon_file_path(best=True)
        logo = QLabel()
        logo.setPixmap(
            QPixmap(logo_file_path).scaled(
                QSize(100, 100),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        layout.addRow(logo)

        self.version = QLabel(f"miniML version {package_meta['version']}")
        layout.addRow(self.version)

        authors = ", ".join(package_meta["authors"]) or ""
        if authors:
            self.author = QLabel(f"Authors: {authors}")
            layout.addRow(self.author)

        urls = package_meta["urls"]
        website_url = urls.get("Repository") or urls.get("Homepage")
        if website_url:
            website_text = "miniML GitHub repository"
            self.website = QLabel(
                f'Website: <a href="{website_url}">{website_text}</a>'
            )
            self.website.setOpenExternalLinks(True)
            layout.addRow(self.website)

        publication_url = urls.get("Publication")
        if publication_url:
            self.paper = QLabel(
                f'Publication: <a href="{publication_url}">{publication_url}</a>'
            )
            self.paper.setOpenExternalLinks(True)
            layout.addRow(self.paper)

        self.setLayout(layout)

        finalize_dialog_window(self, title="About miniML", cancel=False)


class SummaryPanel(QDialog):
    """
    Summary dialog for aggregate detection statistics.
    """

    def __init__(self, *, trace_filename: str, detection, parent=None):
        """
        Build the summary view and populate its fields.
        """
        super().__init__(parent)

        self.populate_fields(trace_filename=trace_filename, detection=detection)
        layout = QFormLayout(self)
        layout.addRow("Filename:", self.filename)
        layout.addRow("Events found:", self.event_count)
        layout.addRow("Events deleted:", self.deleted_event_count)
        layout.addRow("Event frequency (Hz):", self.event_frequency)
        layout.addRow("Average score:", self.average_score)
        layout.addRow(
            f"Average amplitude ({detection.trace.y_unit}):",
            self.average_amplitude,
        )
        layout.addRow(
            f"Median amplitude ({detection.trace.y_unit}):",
            self.median_amplitude,
        )
        layout.addRow("Coefficient of variation:", self.amplitude_cv)
        layout.addRow(f"Average area ({detection.trace.y_unit}*s):", self.average_area)
        layout.addRow("Average risetime (ms):", self.average_rise_time)
        layout.addRow("Average rise slope (pA/ms):", self.average_slope)
        layout.addRow("Average 50% decay time (ms):", self.average_decay_time)
        layout.addRow("Average halfwidth (ms):", self.average_halfwidth)
        layout.addRow("Decay time constant (ms):", self.decay_tau)
        self.setLayout(layout)

        finalize_dialog_window(self, title="Summary", cancel=False)

    def populate_fields(self, *, trace_filename: str, detection):
        """
        Fill summary widgets with computed event statistics.
        """
        self.filename = QLineEdit(trace_filename)
        self.filename.setReadOnly(True)
        self.event_count = QLineEdit(str(detection.event_stats.event_count))
        self.event_count.setReadOnly(True)
        self.deleted_event_count = QLineEdit(str(detection.deleted_events))
        self.deleted_event_count.setReadOnly(True)
        self.event_frequency = QLineEdit(f"{detection.event_stats.frequency():.5f}")
        self.event_frequency.setReadOnly(True)
        self.average_score = QLineEdit(
            f"{detection.event_stats.mean(detection.event_stats.event_scores):.5f}"
        )
        self.average_score.setReadOnly(True)
        self.average_amplitude = QLineEdit(
            f"{detection.event_stats.mean(detection.event_stats.amplitudes):.5f}"
        )
        self.average_amplitude.setReadOnly(True)
        self.median_amplitude = QLineEdit(
            f"{detection.event_stats.median(detection.event_stats.amplitudes):.5f}"
        )
        self.median_amplitude.setReadOnly(True)
        self.amplitude_cv = QLineEdit(
            f"{detection.event_stats.cv(detection.event_stats.amplitudes):.5f}"
        )
        self.amplitude_cv.setReadOnly(True)
        self.average_area = QLineEdit(
            f"{detection.event_stats.mean(detection.event_stats.charges):.5f}"
        )
        self.average_area.setReadOnly(True)
        self.average_rise_time = QLineEdit(
            f"{detection.event_stats.mean(detection.event_stats.risetimes) * 1e3:.5f}"
        )
        self.average_rise_time.setReadOnly(True)
        self.average_slope = QLineEdit(
            f"{detection.event_stats.mean(detection.event_stats.slopes) * 1e-3:.5f}"
        )
        self.average_slope.setReadOnly(True)
        self.average_decay_time = QLineEdit(
            f"{detection.event_stats.mean(detection.event_stats.halfdecays) * 1e3:.5f}"
        )
        self.average_decay_time.setReadOnly(True)
        self.average_halfwidth = QLineEdit(
            f"{detection.event_stats.mean(detection.event_stats.halfwidths) * 1e3:.5f}"
        )
        self.average_halfwidth.setReadOnly(True)
        self.decay_tau = QLineEdit(
            f"{detection.event_stats.mean(detection.event_stats.avg_tau_decay) * 1e3:.5f}"
        )
        self.decay_tau.setReadOnly(True)
