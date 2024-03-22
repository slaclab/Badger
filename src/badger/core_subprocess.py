import typing
import time
import datetime
from .db import load_routine
from pandas import concat, DataFrame
import logging
from badger.errors import (
    BadgerRunTerminatedError,
)
from badger.routine import Routine
from badger.logger import _get_default_logger
from badger.logger.event import Events
from badger.utils import (
    curr_ts_to_str,
    dump_state,
)


def check_critical(self, queue):
        """
        Check if a critical constraint has been violated in the last data point,
        and take appropriate actions if so.

        If there are no critical constraints, the function will return without taking
        any action. If a critical constraint has been violated, it will pause the
        run, open a dialog to inform the user about the violation, and provide
        options to terminate or resume the run.

        Returns
        -------
        None

        Notes
        -----

        The critical constraints are determined by the
        `self.routine.critical_constraint_names` attribute. If no critical
        constraints are defined, this function will have no effect.

        The function emits signals `self.sig_pause` and `self.sig_stop` to handle the
        pause and stop actions.

        """

        # if there are no critical constraints then skip
        if len(self.criticalConstraintNames) == 0:
            return False
        else:
            feas = self.vocs.feasibility_data(self.routine.data.iloc[-1])
            violated_critical = ~feas[self.criticalConstraintNames].any()

            if not violated_critical:
                return False

        queue.put(str(feas))
        return True 


def check_run_status(self, routine, stop_process, pause_process, termination_condition = None):
        """
        check for termination condition
        - checks for internal triggers (max eval, max time) and external triggers
        """
        # Check if termination condition has been satisfied
        if termination_condition:
            tc_config = termination_condition
            idx = tc_config['tc_idx']
            if idx == 0:
                max_eval = tc_config['max_eval']
                if len(routine.data) >= max_eval:
                    stop_process.is_set()

            elif idx == 1:
                max_time = tc_config['max_time']
                dt = time.time() - self.start_time # need to pipe time? 
                if dt >= max_time:
                    stop_process.is_set()

        # External triggers
        if stop_process.is_set():
            raise BadgerRunTerminatedError
        elif pause_process.is_set():
            pause_process.wait()
        else:
            return 0  # continue to run

def convert_to_solution(result: DataFrame, routine: Routine):
    vocs = routine.vocs
    try:
        best_idx, _ = vocs.select_best(routine.sorted_data, n=1)
        if best_idx != len(routine.data) - 1:
            is_optimal = False
        else:
            is_optimal = True
    except NotImplementedError:
        is_optimal = False  # disable the optimal highlight for MO problems

    vars = list(result[vocs.variable_names].to_numpy()[0])
    objs = list(result[vocs.objective_names].to_numpy()[0])
    cons = list(result[vocs.constraint_names].to_numpy()[0])
    stas = list(result[vocs.observable_names].to_numpy()[0])

    solution = (vars, objs, cons, stas, is_optimal,
                vocs.variable_names,
                vocs.objective_names,
                vocs.constraint_names,
                vocs.observable_names)

    return solution


def run_routine_subprocess(queue, evaluate_queue, stop_process, pause_process, wait_event) -> None:
    """
    Run the provided routine object using Xopt. This method is run as a subproccess
    Parameters
    ----------
    queue : 
        
    stop_process : 
    pause_process : 
    """
    wait_event.wait()
    
    try:
        args = queue.get(timeout=1)
    except Exception as e:
        print(f"Error in subprocess: {type(e).__name__}, {str(e)}")

    # set required arguments 
    routine, _ = load_routine(args['routine_name'])

    # routine_queue.put(routine.vocs)

    # set optional arguments 
    try:
        evaluate = args['evaluate']
    except KeyError:
        evaluate = None 

    try:
        save_states = args['save_states']
    except KeyError:
        save_states = None 

    try:
        dump_file_callback = args['dump_file_callback']
    except KeyError:
        dump_file_callback = None

    try:
        verbose = args['verbose']
    except KeyError:
        verbose = 2

    try:
        termination_condition = args['termination_condition']
    except KeyError:
        termination_condition = None

    environment = routine.environment
    initial_points = routine.initial_points

    # Log the optimization progress in terminal
    opt_logger = _get_default_logger(verbose)

    # Save system states if applicable
    states = environment.get_system_states()
    if save_states and (states is not None):
        queue.put(states) # might need to change queue here 

    # Optimization starts
    solution_meta = (None, None, None, None, None,
                     routine.vocs.variable_names,
                     routine.vocs.objective_names,
                     routine.vocs.constraint_names,
                     routine.vocs.observable_names)
    opt_logger.update(Events.OPTIMIZATION_START, solution_meta)

    # evaluate initial points:
    # Nikita: more care about the setting var logic,
    # wait or consider timeout/retry
    # TODO: need to evaluate a single point at the time

    for _, ele in initial_points.iterrows():
        result = routine.evaluate_data(ele.to_dict())
        solution = convert_to_solution(result, routine)
        opt_logger.update(Events.OPTIMIZATION_STEP, solution)
        if evaluate:
            evaluate_queue.put(result)

    # Prepare for dumping file
    if dump_file_callback:
        combined_results = None
        ts_start = curr_ts_to_str()
        dump_file = dump_file_callback()
        if not dump_file:
            dump_file = f"xopt_states_{ts_start}.yaml"

    # perform optimization
    try:
        while True:

            if stop_process.is_set():
                raise BadgerRunTerminatedError
            elif pause_process.is_set():
                pause_process.wait() 

            # generate points to observe
            candidates = routine.generator.generate(1)[0]
            candidates = DataFrame(candidates, index=[0])

            # generate_callback(generator, candidates)
            # generate_callback(candidates)

            if stop_process.is_set():
                raise BadgerRunTerminatedError
            elif pause_process.is_set():
                pause_process.wait() 

            # if still active evaluate the points and add to generator
            # check active_callback evaluate point
            result = routine.evaluate_data(candidates)
            solution = convert_to_solution(result, routine)
            opt_logger.update(Events.OPTIMIZATION_STEP, solution)
            if evaluate:
                evaluate_queue.put(routine.data)

            # Dump Xopt state after each step
            if dump_file_callback:
                if combined_results is not None:
                    combined_results = concat([combined_results, result],
                                              axis=0).reset_index(drop=True)
                else:
                    combined_results = result

                dump_state(dump_file, routine.generator, combined_results)
    except Exception as e:
        opt_logger.update(Events.OPTIMIZATION_END, solution_meta)
        raise e