from abc import abstractmethod, ABC

import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLineEdit, QPushButton, QWidget, QScrollArea, QFormLayout,
    QComboBox, QCheckBox, QLabel, QGridLayout, QSizePolicy
)
from badger.core import Routine


class AnalysisExtension(QDialog):
    window_closed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent=parent)

    @abstractmethod
    def update_window(self, routine: Routine):
        pass

    def closeEvent(self, event) -> None:
        self.window_closed.emit(self)
        super().closeEvent(event)

# Class for Xopt, pass in AnalysisExtension

class XoptVisualizer(AnalysisExtension):
    def __init__(self, parent=None, max_plot_vars=2, xopt_obj=None):
        super().__init__(parent)

        # Ignore all warnings
        import warnings
        warnings.filterwarnings("ignore")
        import time
        import math

        from xopt.vocs import VOCS

        # define variables, function objective and constraining function
        vocs = VOCS(
            variables={"x0": [0., 2. * math.pi], "x1":[0., 2.*math.pi]},
            objectives={"f": "MINIMIZE"},
            constraints={"c": ["LESS_THAN", 0]}
        )

        # Define visualization options
        visualization_options = {
            "output_names": None,
            "variable_names": None,
            "idx": -1,
            "reference_point": None,
            "show_samples": True,
            "show_prior_mean": False,
            "show_feasibility": False,
            "show_acquisition": True,
            "n_grid": 50,
            "axes": None,
        }

        # define a test function to optimize
        import numpy as np

        def test_function(input_dict):
            return {"f": np.sin(input_dict["x0"]*input_dict["x1"]),"c": np.cos
            (input_dict["x0"])}


        # ## Create Xopt objects
        # Create the evaluator to evaluate our test function and create a generator that uses
        # the Expected Improvement acquisition function to perform Bayesian Optimization.


        from xopt.evaluator import Evaluator
        from xopt.generators.bayesian import ExpectedImprovementGenerator
        from xopt import Xopt

        evaluator = Evaluator(function=test_function)
        generator = ExpectedImprovementGenerator(vocs=vocs)
        X = Xopt(evaluator=evaluator, generator=generator, vocs=vocs)

        X.random_evaluate(n_samples=5)
        X.generator.train_model()

        self.initUI(X, vocs, visualization_options, max_plot_vars, xopt_obj)

    def initUI(self, X, vocs, visualization_options, max_plot_vars, xopt_obj):
        self.setWindowTitle('Xopt Visualizer')

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas # lazy import
        from matplotlib.figure import Figure # lazy import
        
        # Use provided Xopt object, or default to X from the example
        self.X = xopt_obj if xopt_obj else X
        
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.max_plot_vars = max_plot_vars  # Maximum number of variables to plot
        
        layout = QVBoxLayout()
        
        # Grid layout for variables, references, and checkboxes
        self.grid_layout = QGridLayout()
        self.variable_dropdowns = []
        self.ref_inputs = []
        self.include_checkboxes = []

        for i, var_name in enumerate(vocs.variable_names):
            # Variable dropdown
            var_dropdown = QComboBox()
            var_dropdown.addItems(vocs.variable_names)
            var_dropdown.setCurrentText(vocs.variable_names[i])
            self.variable_dropdowns.append(var_dropdown)
            self.grid_layout.addWidget(QLabel(f"Variable {i+1}:"), i, 0)
            self.grid_layout.addWidget(var_dropdown, i, 1)

            # Reference input
            ref_input = QLineEdit("0.0")
            self.ref_inputs.append(ref_input)
            self.grid_layout.addWidget(QLabel("Reference:"), i, 2)
            self.grid_layout.addWidget(ref_input, i, 3)

            # Include checkbox 
            include_checkbox = QCheckBox(f"Include Variable {i+1}")
            include_checkbox.stateChanged.connect(self.update_ref_inputs)  # Connect to update function
            self.include_checkboxes.append(include_checkbox)
            self.grid_layout.addWidget(include_checkbox, i, 4)
        
        # Initialize checkboxes to be unchecked
        for checkbox in self.include_checkboxes:
            checkbox.setChecked(False)
            checkbox.stateChanged.connect(self.enforce_checkbox_rule)

        self.last_unchecked = None  # Track the last unchecked checkbox

        layout.addLayout(self.grid_layout)

        # Add input fields and checkboxes for visualization options
        self.param_widgets = []
        for i, (key, value) in enumerate(visualization_options.items()):
            name_input = QLineEdit(key)
            value_input = QLineEdit(str(value))
            include_checkbox = QCheckBox(f"Include {key}")
            self.param_widgets.append((name_input, value_input, include_checkbox))
            self.grid_layout.addWidget(QLabel(f"{key}:"), i + len(vocs.variable_names), 0)
            self.grid_layout.addWidget(value_input, i + len(vocs.variable_names), 1)
            self.grid_layout.addWidget(include_checkbox, i + len(vocs.variable_names), 2)

        # Checkbox for acquisition function
        self.show_acq_checkbox = QCheckBox("Show Acquisition Function")
        layout.addWidget(self.show_acq_checkbox)

        # Update button
        update_button = QPushButton("Update Plot")
        update_button.clicked.connect(self.update_plot)
        layout.addWidget(update_button)

        # Plot area
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.update_ref_inputs()  # Initialize reference inputs based on checkboxes

    def handle_dropdown_change(self, variable_names):
        # Get the current selections
        selected_0 = self.variable_dropdowns[0].currentText()
        selected_1 = self.variable_dropdowns[1].currentText()

        # Update the other dropdown based on the current selection
        if self.sender() == self.variable_dropdowns[0]:
            if selected_0 == variable_names[0]:
                self.variable_dropdowns[1].setCurrentText(variable_names[1])
            else:
                self.variable_dropdowns[1].setCurrentText(variable_names[0])
        elif self.sender() == self.variable_dropdowns[1]:
            if selected_1 == variable_names[0]:
                self.variable_dropdowns[0].setCurrentText(variable_names[1])
            else:
                self.variable_dropdowns[0].setCurrentText(variable_names[0])

    def enforce_checkbox_rule(self, state):
        # Only consider the two variables from vocs.variable_names
        if len(self.include_checkboxes) != 2:
            return

        sender = self.sender()
        other_checkbox = next(cb for cb in self.include_checkboxes if cb != sender)

        if sender.isChecked():
            other_checkbox.setChecked(False)
            other_checkbox.setEnabled(False)
        else:
            other_checkbox.setEnabled(True)

        self.update_ref_inputs()

    def update_ref_inputs(self):
        for i, ref_input in enumerate(self.ref_inputs):
            if self.include_checkboxes[i].isChecked():
                ref_input.setEnabled(False)
                ref_input.hide()
            else:
                ref_input.setEnabled(True)
                ref_input.show()

    def update_plot(self):
        # Get the selected variable names from the dropdowns
        variable_names = [dropdown.currentText() for dropdown, checkbox in zip(self.variable_dropdowns, self.include_checkboxes) if checkbox.isChecked()]
        variable_names = [var for var in variable_names if var != ""]  # Filter out empty selections

        # Check if the same variable is selected in both dropdowns
        if len(set(variable_names)) != len(variable_names):
            print("Error: The same variable cannot be selected for both dropdowns.")
            return

        # Ensure the number of selected variables is either 1 or 2
        if len(variable_names) > 2:
            print("Error: Visualization is only supported with respect to 1 or 2 variables.")
            return
        
        print("Selected variables:", variable_names)
        # Limit to a maximum of two variables
        variable_names = variable_names[:self.max_plot_vars] 

        # Create reference_point dictionary for non-selected variables only
        reference_point = {}
        for i, var in enumerate(self.X.vocs.variable_names):
            if var not in variable_names:
                ref_value = float(self.ref_inputs[i].text())
                reference_point[var] = ref_value

        # Read the input values and create a kwargs dictionary for checked options
        kwargs = {}
        for name_input, value_input, include_checkbox in self.param_widgets:
            if include_checkbox.isChecked():
                name = name_input.text()
                value = eval(value_input.text())
                kwargs[name] = value

        # Add variable_names and reference_point to kwargs if they are checked
        if variable_names:
            kwargs['variable_names'] = variable_names
            kwargs['reference_point'] = reference_point
            
        # Print kwargs for debugging
        print("kwargs:", kwargs)

        self.figure.clear()

        from xopt.generators.bayesian.visualize import visualize_generator_model # lazy import

        try:
            fig, ax = visualize_generator_model(
                self.X.generator,
                **kwargs
            )
        except ValueError as e:
            print(f"Error: {e}")
            return

        self.canvas.figure = fig
        self.canvas.draw()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.canvas.resize(self.canvas.sizeHint())
        self.updateGeometry()
        self.resize(self.sizeHint())


class ParetoFrontViewer(AnalysisExtension):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.setWindowTitle("Pareto Front Viewer")

        self.plot_widget = pg.PlotWidget()

        self.scatter_plot = self.plot_widget.plot(pen=None, symbol='o', symbolSize=10)

        layout = QVBoxLayout()
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

    def update_window(self, routine: Routine):
        if len(routine.vocs.objective_names) != 2:
            raise ValueError("cannot use pareto front viewer unless there are 2 "
                             "objectives")

        x_name = routine.vocs.objective_names[0]
        y_name = routine.vocs.objective_names[1]

        if routine.data is not None:
            x = routine.data[x_name]
            y = routine.data[y_name]

            # Update the scatter plot
            self.scatter_plot.setData(x=x, y=y)

        # set labels
        self.plot_widget.setLabel("left", y_name)
        self.plot_widget.setLabel("bottom", x_name)
