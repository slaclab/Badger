import traceback

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, \
    QWidget, QLabel
from badger.gui.default.components.analysis_extensions import AnalysisExtension, \
    ParetoFrontViewer


class ExtensionsPalette(QMainWindow):
    """
    A QMainWindow-based user interface for managing and launching extensions in Badger.

    Parameters
    ----------
    run_monitor : RunMonitor
        The run monitor associated with the palette.

    Attributes
    ----------
    base_text : str
        Base text for the number of active extensions.
    text_box : QLabel
        QLabel widget for displaying the number of active extensions.
    btn_data_viewer : QPushButton
        QPushButton for launching the ParetoFrontViewer extension.

    Methods
    -------
    n_active_extensions
        Property to get the number of active extensions.
    update_palette
        Update the display of the active extensions count.
    add_pf_viewer
        Open the ParetoFrontViewer extension.
    add_child_window_to_monitor(child_window)
        Add a child window to the run monitor.

    """
    def __init__(self, run_monitor):
        """
        Initialize the ExtensionsPalette.

        Parameters
        ----------
        run_monitor : RunMonitor
            The run monitor associated with the palette.

        """
        super(ExtensionsPalette, self).__init__()

        self.run_monitor = run_monitor

        self.setWindowTitle('Badger Extensions Palette')
        self.setGeometry(100, 100, 200, 200)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        self.base_text = "Number of active exensions: "
        self.text_box = QLabel(self.base_text + "0", self)

        self.btn_data_viewer = QPushButton('ParetoFrontViewer')

        layout.addWidget(self.text_box)
        layout.addWidget(self.btn_data_viewer)

        central_widget.setLayout(layout)

        self.btn_data_viewer.clicked.connect(self.add_pf_viewer)

    @property
    def n_active_extensions(self):
        """
        Property to get the number of active extensions.

        Returns
        -------
        int
            The number of active extensions.

        """
        return len(self.run_monitor.active_extensions)

    def update_palette(self):
        self.text_box.setText(
            self.base_text + str(self.n_active_extensions))

    def add_pf_viewer(self):
        """
        Open the ParetoFrontViewer extension.

        """
        self.add_child_window_to_monitor(ParetoFrontViewer())

    def add_child_window_to_monitor(self, child_window: AnalysisExtension):
        """
        Add a child window to the run monitor.

        Parameters
        ----------
        child_window : AnalysisExtension
            The child window (extension) to add to the run monitor.

        """
        child_window.show()
        child_window.window_closed.connect(self.run_monitor.extension_window_closed)
        self.run_monitor.active_extensions.append(child_window)

        if self.run_monitor.routine is not None:
            try:
                self.run_monitor.active_extensions[-1].update_window(
                    self.run_monitor.routine
                )
            except ValueError:
                traceback.print_exc()

        self.update_palette()

