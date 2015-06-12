cimport numpy as np
import numpy as np
from libcpp.string cimport string
cimport _wrapper
from ..utils.log import HideExtern


cdef class CppBLLoader:
    """Behavior learning loader.

    This is the Python wrapper around the C++ library.
    """
    cdef BLLoader *thisptr

    def __cinit__(self):
        with HideExtern("stderr"):
            self.thisptr = new BLLoader()

    def __dealloc__(self):
        with HideExtern("stderr"):
            del self.thisptr

    def load_library(self, lib_path):
        """Load library from a file.

        Parameters
        ----------
        lib_path : str
            Path to the library file for the libManager.
        """
        cdef char* library_file_cstr = lib_path
        with HideExtern("stderr"):
            self.thisptr.loadLibrary(string(library_file_cstr),NULL)

    def load_config_file(self, config_file):
        """Load list of libraries for the libManager from a file.

        Parameters
        ----------
        config_file : str
            Path to the configuration file for the libManager. The file
            should contain a list of shared libraries that can be loaded
            by the libManager.
        """
        cdef char* config_file_cstr = config_file
        with HideExtern("stderr"):
            self.thisptr.loadConfigFile(string(config_file_cstr))

    def acquire_optimizer(self, name):
        """Get an optimizer.

        Parameters
        ----------
        name : str
            Name of the corresponding shared library.

        Returns
        -------
        optimizer : CppOptimizer
            Optimizer instance.
        """
        cdef char* name_cstr = name
        optimizer = CppOptimizer()
        with HideExtern("stderr"):
            optimizer.thisptr = self.thisptr.acquireOptimizer(string(name_cstr))
        return optimizer

    def acquire_behavior_search(self, name):
        """Get a behavior search.

        Parameters
        ----------
        name : str
            Name of the corresponding shared library.

        Returns
        -------
        optimizer : CppBehaviorSearch
            Behavior search instance.
        """
        cdef char* name_cstr = name
        behavior_search = CppBehaviorSearch()
        with HideExtern("stderr"):
            behavior_search.thisptr = self.thisptr.acquireBehaviorSearch(string(name_cstr))
        return behavior_search

    def acquire_environment(self, name):
        """Get an environment.

        Parameters
        ----------
        name : str
            Name of the corresponding shared library.

        Returns
        -------
        optimizer : CppEnvironment
            Environment instance.
        """
        cdef char* name_cstr = name
        environment = CppEnvironment()
        with HideExtern("stderr"):
            environment.thisptr = self.thisptr.acquireEnvironment(string(name_cstr))
        return environment


    def acquire_behavior(self, name):
        """Get a behavior.

        Parameters
        ----------
        name : str
            Name of the corresponding shared library.

        Returns
        -------

        behavior : CppLoadableBehavior
            Behavior instance.
        """
        cdef char* name_cstr = name
        behavior = CppLoadableBehavior()
        with HideExtern("stderr"):
            behavior.behaviorPtr = self.thisptr.acquireBehavior(name)
            behavior.thisptr = behavior.behaviorPtr

        return behavior

    def acquire_contextual_environment(self, name):
        """Get an contextual environment.

        name : CppContextualEnvironment
            ContextualEnvironment instance.
        """
        cdef char* name_cstr = name
        environment = CppContextualEnvironment()
        with HideExtern("stderr"):
            environment.thisptr = self.thisptr.acquireContextualEnvironment(string(name_cstr))
        return environment

    def release_library(self, name):
        """Release a C++ library."""
        cdef char* name_cstr = name
        with HideExtern("stderr"):
            self.thisptr.releaseLibrary(string(name_cstr))


cdef class CppOptimizer:
    cdef Optimizer *thisptr

    def __cinit__(self):
        self.thisptr = NULL  # The BLLoader will delete this pointer

    def init(self, dimension):
        """Initialize optimizer.

        Parameters
        ----------
        dimension : int
            dimension of the parameter vector
        """
        self.thisptr.init(dimension)

    def get_next_parameters(self, p):
        """Get next individual/parameter vector for evaluation.

        Parameters
        ----------
        p : array_like, shape (num_p,)
            parameter vector, will be modified
        """
        assert(p.ndim == 1)
        cdef np.ndarray[double, ndim=1, mode="c"] params_array = np.ndarray(*p.shape)
        self.thisptr.getNextParameters(&params_array[0], params_array.shape[0])
        p[:] = params_array

    def get_best_parameters(self, p):
      """Get best individual/parameter vector for evaluation.

      Parameters
      ----------
      p : array_like, shape (num_p,)
      parameter vector, will be modified
      """
      assert(p.ndim == 1)
      cdef np.ndarray[double, ndim=1, mode="c"] params_array = np.ndarray(*p.shape)
      self.thisptr.getBestParameters(&params_array[0], params_array.shape[0])
      p[:] = params_array

    def set_evaluation_feedback(self, rewards):
        """Set feedbacks for the parameter vector.

        Parameters
        ----------
        rewards : list of float
            feedbacks for each step or for the episode, depends on the problem
        """
        assert(rewards.ndim == 1)
        cdef np.ndarray[double, ndim=1, mode="c"] rewards_array = np.ndarray(*rewards.shape)
        rewards_array[:] = rewards
        self.thisptr.setEvaluationFeedback(&rewards_array[0], rewards_array.shape[0])

    def is_behavior_learning_done(self):
        """Check if the behavior learning is finished.

        Returns
        -------
        finished : bool
            Is the learning of a behavior finished?
        """
        return self.thisptr.isBehaviorLearningDone()

    def get_best_parameters(self):
        """Return best parameters found so far by Optimizer.

        Returns
        -------
        parameters : array_like, shape = (num_p,)
            best parameters found so far
        """
        # TODO add this function to C++ interface
        raise NotImplementedError("Cannot obtain best parameters from C++ optimizer!")

cdef class CppBehaviorSearch:
    cdef BehaviorSearch *thisptr

    def __cinit__(self):
        self.thisptr = NULL  # The BLLoader will delete this pointer

    def init(self, num_inputs, num_outputs):
        """Initialize the behavior search.

        Parameters
        ----------
        num_inputs : int
            number of inputs of the behavior
        num_outputs : int
            number of outputs of the behavior
        """
        self.thisptr.init(num_inputs, num_outputs)

    def get_next_behavior(self):
        """Obtain next behavior for evaluation.

        Returns
        -------
        behavior : Behavior
            mapping from input to output
        """
        behavior = CppBehavior()
        behavior.thisptr = self.thisptr.getNextBehavior()
        return behavior

    def get_best_behavior(self):
        """Returns the best behavior found so far.

        Returns
        -------
        behavior : Behavior
            mapping from input to output
        """
        # TODO add this function to C++ interface
        raise NotImplementedError("Cannot obtain best behavior from C++ behavior search")

    def set_evaluation_feedback(self, feedbacks):
        """Set feedback for the last behavior.

        Parameters
        ----------
        feedbacks : list of float
            feedback for each step or for the episode, depends on the problem
        """
        assert(feedbacks.ndim == 1)
        cdef np.ndarray[double, ndim=1, mode="c"] feedbacks_array = np.ndarray(*feedbacks.shape)
        feedbacks_array[:] = feedbacks
        self.thisptr.setEvaluationFeedback(&feedbacks_array[0], feedbacks_array.shape[0])

    def write_results(self, result_path):
        """Store current search state.

        Parameters
        ----------
        result_path : string
            path in which the state should be stored
        """
        cdef char* result_path_cstr = result_path
        self.thisptr.writeResults(string(result_path_cstr))

    def get_behavior_from_results(self, result_path):
        """Recover search state from file.

        Parameters
        ----------
        result_path : string
            path in which we search for the file
        """
        cdef char* result_path_cstr = result_path
        behavior = CppBehavior()
        behavior.thisptr = self.thisptr.getBehaviorFromResults(string(result_path_cstr))
        return behavior


cdef class CppEnvironment:
    cdef Environment *thisptr

    def __cinit__(self):
        self.thisptr = NULL  # The BLLoader will delete this pointer

    def init(self):
        """Initialize environment."""
        self.thisptr.init()

    def reset(self):
        """Reset state of the environment."""
        self.thisptr.reset()

    def get_num_inputs(self):
        """Get number of environment inputs.

        Parameters
        ----------
        n : int
            number of environment inputs
        """
        return self.thisptr.getNumInputs()

    def get_num_outputs(self):
        """Get number of environment outputs.

        Parameters
        ----------
        n : int
            number of environment outputs
        """
        return self.thisptr.getNumOutputs()

    def get_outputs(self, values):
        """Get environment outputs, e.g. state of the environment.

        Parameters
        ----------
        values : array
            outputs for the environment, will be modified
        """
        cdef int n_outputs = self.get_num_outputs()
        cdef np.ndarray[double, ndim=1, mode="c"] outputs = np.ndarray(n_outputs)
        self.thisptr.getOutputs(&outputs[0], n_outputs)
        values[:] = outputs

    def set_inputs(self, values):
        """Set environment inputs, e.g. next action.

        Parameters
        ----------
        values : array,
            input of the environment
        """
        cdef int n_inputs = self.get_num_inputs()
        cdef np.ndarray[double, ndim=1, mode="c"] inputs = np.ndarray(n_inputs)
        inputs[:] = values
        self.thisptr.setInputs(&inputs[0], n_inputs)

    def step_action(self):
        """Take a step in the environment.
        """
        self.thisptr.stepAction()

    def is_evaluation_done(self):
        """Check if the evaluation of the behavior is finished.

        Returns
        -------
        finished : bool
            Is the evaluation finished?
        """
        return self.thisptr.isEvaluationDone()

    def get_feedback(self):
        """Get the feedbacks for the last evaluation period.

        Returns
        -------
        feedback : array
            Feedback values
        """
        cdef np.ndarray[double, ndim=1, mode="c"] feedback = np.ndarray(1000)
        n_feedbacks = self.thisptr.getFeedback(&feedback[0])
        if n_feedbacks > 1000:
            raise ValueError("Collected more than 1000 feedbacks, fix the "
                             "wrapper code")
        return feedback[:n_feedbacks]

    def is_behavior_learning_done(self):
        """Check if the behavior learning is finished.

        Returns
        -------
        finished : bool
            Is the learning of a behavior finished?
        """
        return self.thisptr.isBehaviorLearningDone()

    def is_contextual(self):
        """Return whether environment is contextual

        Returns
        -------
        contextual : bool
            whether this environment is contextual
        """
        return self.thisptr.isContextual()


cdef class CppContextualEnvironment:
    cdef ContextualEnvironment *thisptr

    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        del self.thisptr

    def init(self):
        """Initialize environment."""
        self.thisptr.init()

    def reset(self):
        """Reset state of the environment."""
        self.thisptr.reset()

    def get_num_inputs(self):
        """Get number of environment inputs.

        Parameters
        ----------
        n : int
            number of environment inputs
        """
        return self.thisptr.getNumInputs()

    def get_num_outputs(self):
        """Get number of environment outputs.

        Parameters
        ----------
        n : int
            number of environment outputs
        """
        return self.thisptr.getNumOutputs()

    def get_outputs(self, values):
        """Get environment outputs, e.g. state of the environment.

        Parameters
        ----------
        values : array
            outputs for the environment, will be modified
        """
        cdef int n_outputs = self.get_num_outputs()
        cdef np.ndarray[double, ndim=1, mode="c"] outputs = np.ndarray(n_outputs)
        self.thisptr.getOutputs(&outputs[0], n_outputs)
        values[:] = outputs

    def set_inputs(self, values):
        """Set environment inputs, e.g. next action.

        Parameters
        ----------
        values : array,
            input of the environment
        """
        cdef int n_inputs = self.get_num_inputs()
        cdef np.ndarray[double, ndim=1, mode="c"] inputs = np.ndarray(n_inputs)
        inputs[:] = values
        self.thisptr.setInputs(&inputs[0], n_inputs)

    def step_action(self):
        """Take a step in the environment.
        """
        self.thisptr.stepAction()

    def is_evaluation_done(self):
        """Check if the evaluation of the behavior is finished.

        Returns
        -------
        finished : bool
            Is the evaluation finished?
        """
        return self.thisptr.isEvaluationDone()

    def get_feedback(self):
        """Get the feedbacks for the last evaluation period.

        Returns
        -------
        feedback : array
            Feedback values
        """
        cdef np.ndarray[double, ndim=1, mode="c"] feedback = np.ndarray(1000)
        n_feedbacks = self.thisptr.getFeedback(&feedback[0])
        if n_feedbacks > 1000:
            raise ValueError("Collected more than 1000 feedbacks, fix the "
                             "wrapper code")
        return feedback[:n_feedbacks]

    def is_behavior_learning_done(self):
        """Check if the behavior learning is finished.

        Returns
        -------
        finished : bool
            Is the learning of a behavior finished?
        """
        return self.thisptr.isBehaviorLearningDone()

    def is_contextual(self):
        """Return whether environment is contextual

        Returns
        -------
        contextual : bool
            whether this environment is contextual
        """
        return self.thisptr.isContextual()

    def request_context(self, context):
        """Sets the context of the environment to the given one

        Parameters
        ----------
        context : array
            the requested context

        Returns
        -------
        contex : double*
            the context the environment was set to
        """
        cdef int n_dims = self.get_num_context_dims()
        cdef double *tmp_array
        cdef np.ndarray[double, ndim=1, mode="c"] context_out  = np.ndarray(n_dims)
        cdef np.ndarray[double, ndim=1, mode="c"] context_in  = np.ndarray(n_dims)
        context_in[:] = context
        tmp_array = self.thisptr.request_context(&context_in[0], n_dims)
        for i in range(n_dims):
            context_out[i] = tmp_array[i]
        return context_out[:]

    def get_num_context_dims(self):
        """returns the count of the context dimensions

        Returns
        -------
        contex_dimensions : int
            the dimension of the context
        """
        return self.thisptr.get_num_context_dims()

cdef class CppBehavior:
    cdef Behavior *thisptr

    def __cinit__(self):
        self.thisptr = NULL  # The BLLoader will delete this pointer

    def set_meta_parameters(self, keys, meta_parameters):
        """Set meta-parameters.

        Meta-parameters could be the goal, obstacles, ...

        Parameters
        ----------
        keys : list of string
            names of meta-parameters
        meta_parameters : list of float
            values of meta-parameters
        """
        raise NotImplementedError("Meta parameters are not available in C++ behavior!")

    def set_inputs(self, values):
        """Set input for the next step.

        Parameters
        ----------
        inputs : array-like, shape = [num_inputs,]
            inputs, e.g. current state of the system
        """
        cdef int n_inputs = values.shape[0]
        cdef np.ndarray[double, ndim=1, mode="c"] inputs = np.ndarray(n_inputs)
        inputs[:] = values
        self.thisptr.setInputs(&inputs[0], n_inputs)

    def get_outputs(self, values):
        """Get outputs of the last step.

        Parameters
        ----------
        outputs : array-like, shape = [num_outputs,]
            outputs, e.g. next action, will be updated
        """
        cdef int n_outputs = values.shape[0]
        cdef np.ndarray[double, ndim=1, mode="c"] outputs = np.ndarray(n_outputs)
        self.thisptr.getOutputs(&outputs[0], n_outputs)
        values[:] = outputs

    def step(self):
        """Compute output for the received input.

        Use the inputs and meta-parameters to compute the outputs.
        """
        self.thisptr.step()

    def can_step(self):
        """Determine if stepping is possible.

        If this method returns true step() can be called at least once more.
        """
        return self.thisptr.canStep()


cdef class CppLoadableBehavior(CppBehavior):
    cdef LoadableBehavior *behaviorPtr

    def __cinit__(self):
        self.behaviorPtr = NULL  # The BLLoader will delete this pointer

    def initialize(self, initialConfigPath):
        """Initializes the Behavior using the specified config file.

        Initialization should be done only once. It is intended to set the
        Behavior's template parameters.

        The Behavior needs to be initialized before any other method can be called.

        Returns
        -------
        success : bool
            Whether the initialization worked or not
        """
        return self.behaviorPtr.initialize(initialConfigPath)

    def configure(self, configPath):
        """Configures the Behavior using the specified config file.

        This method should be called after initialize().
        It sets the runtime parameters of the Behavior and may be called
        multiple times during execution.

        The config file should be in yaml format.

        Returns
        -------
        success : bool
            Whether the configuration worked or not
        """
        return self.behaviorPtr.configure(configPath)

    def configure_yaml(self, yaml_str):
        """Configures the Behavior using the specified yaml string.

        This method should be called after initialize().
        It sets the runtime parameters of the Behavior and may be called
        multiple times during execution.

        Returns
        -------
        success : bool
            Whether the configuration worked or not
        """
        return self.behaviorPtr.configureYaml(yaml_str)
