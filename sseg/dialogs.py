from __future__ import annotations

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QDialog, QLabel, QPushButton, QSlider, QVBoxLayout


class ParameterDialog(QDialog):
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("Adjust Parameters")
        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(1500)
        self._apply_timer.timeout.connect(self.apply_changes)

        layout = QVBoxLayout(self)

        # Auto 3D opacity is currently deprecated because recent automatic
        # transfer-function heuristics were too aggressive and could render
        # brains fully transparent. Keep the attribute on the viewer for
        # backwards compatibility, but hide the control from the dialog and
        # default to manual settings instead.
        self.steepness_slider = self.create_slider("Steepness", 1, 100, self.viewer.steepness, layout)
        self.exponent_slider = self.create_slider("Exponent", 1, 10, self.viewer.exponent, layout)
        self.opacity_multiplier_slider = self.create_slider(
            "Opacity Multiplier", 1, 400, self.viewer.opacity_multiplier, layout
        )

        reset_button = QPushButton("Reset", self)
        reset_button.clicked.connect(self.reset_parameters)
        layout.addWidget(reset_button)

        self.update_manual_controls_enabled()

    def create_slider(self, label, min_val, max_val, init_val, layout):
        label_widget = QLabel(f"{label}: {init_val}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(init_val)
        slider.valueChanged.connect(lambda value, lbl=label_widget, l=label: self.on_slider_changed(lbl, l, value))
        layout.addWidget(label_widget)
        layout.addWidget(slider)
        slider._value_label = label_widget
        slider._label_name = label
        return slider

    def on_slider_changed(self, label_widget, label_name, value):
        label_widget.setText(f"{label_name}: {value}")
        self._apply_timer.start()

    def update_manual_controls_enabled(self):
        self.steepness_slider.setEnabled(True)
        self.exponent_slider.setEnabled(True)
        self.opacity_multiplier_slider.setEnabled(True)

    def apply_changes(self):
        self.viewer.auto_3d_opacity = False
        self.viewer.steepness = self.steepness_slider.value()
        self.viewer.exponent = self.exponent_slider.value()
        self.viewer.opacity_multiplier = self.opacity_multiplier_slider.value()
        self.viewer.update_visualization(reset_volume=True)

    def reset_parameters(self):
        self.viewer.steepness = self.viewer.default_steepness
        self.viewer.exponent = self.viewer.default_exponent
        self.viewer.opacity_multiplier = self.viewer.default_opacity_multiplier
        self.steepness_slider.setValue(self.viewer.steepness)
        self.exponent_slider.setValue(self.viewer.exponent)
        self.opacity_multiplier_slider.setValue(self.viewer.opacity_multiplier)
        self._apply_timer.start()


class RotationDialog(QDialog):

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("Rotate Image Planes")
        layout = QVBoxLayout()

        self.sliders = {}
        self.labels = {}
        axes = ["x", "y", "z"]
        for axis in axes:
            layout.addWidget(QLabel(f"{axis.upper()} axis rotation:"))
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-90)
            slider.setMaximum(90)
            slider.setValue(0)
            slider.sliderReleased.connect(lambda a=axis, s=slider: self.apply_rotation(a, s))
            label = QLabel("0°")
            label.setAlignment(Qt.AlignCenter)
            self.sliders[axis] = slider
            self.labels[axis] = label
            layout.addWidget(slider)
            layout.addWidget(label)
            slider.valueChanged.connect(lambda value, a=axis: self.update_label(value, a))

        self.reset_button = QPushButton("Reset Rotation")
        self.reset_button.clicked.connect(self.reset_rotation)
        layout.addWidget(self.reset_button)

        self.setLayout(layout)
        self.update_timer = QTimer()
        self.update_timer.setInterval(100)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.reset_all_sliders)

    def apply_rotation(self, axis, slider):
        if self.update_timer.isActive():
            self.update_timer.stop()
        angle = slider.value()
        self.viewer.update_rotation(angle, axis)
        slider.setValue(0)

    def update_label(self, value, axis):
        self.labels[axis].setText(f"{value}°")

    def reset_rotation(self):
        for axis in self.sliders:
            self.sliders[axis].setValue(0)
        self.viewer.reset_rotation()

    def reset_all_sliders(self):
        for _, slider in self.sliders.items():
            slider.setValue(0)

    def slider_changed(self):
        if not self.update_timer.isActive():
            self.update_timer.start()
