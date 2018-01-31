from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
import numpy as np
from cython.operator cimport dereference as deref
cimport _declarations as cpp


cdef class TrajectoryData:
    cdef cpp.TrajectoryData * thisptr
    cdef bool delete_thisptr

    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = True

    def __dealloc__(self):
        if self.delete_thisptr and self.thisptr != NULL:
            del self.thisptr

    def __init__(TrajectoryData self, int numBF, int numDim, bool isStroke, double overlap):
        cdef int cpp_numBF = numBF
        cdef int cpp_numDim = numDim
        cdef bool cpp_isStroke = isStroke
        cdef double cpp_overlap = overlap
        self.thisptr = new cpp.TrajectoryData(cpp_numBF, cpp_numDim, cpp_isStroke, cpp_overlap)

    mean_ = property(get_mean_, set_mean_)

    cpdef get_mean_(TrajectoryData self):
        cdef vector[double] result = self.thisptr.mean_
        return result

    cpdef set_mean_(TrajectoryData self, object mean_):
        cdef vector[double] cpp_mean_ = mean_
        self.thisptr.mean_ = cpp_mean_

    covariance_ = property(get_covariance_, set_covariance_)

    cpdef get_covariance_(TrajectoryData self):
        cdef vector[double] result = self.thisptr.covariance_
        return result

    cpdef set_covariance_(TrajectoryData self, object covariance_):
        cdef vector[double] cpp_covariance_ = covariance_
        self.thisptr.covariance_ = cpp_covariance_

    num_b_f_ = property(get_num_b_f_, set_num_b_f_)

    cpdef get_num_b_f_(TrajectoryData self):
        cdef int result = self.thisptr.numBF_
        return result

    cpdef set_num_b_f_(TrajectoryData self, int numBF_):
        cdef int cpp_numBF_ = numBF_
        self.thisptr.numBF_ = cpp_numBF_

    num_dim_ = property(get_num_dim_, set_num_dim_)

    cpdef get_num_dim_(TrajectoryData self):
        cdef int result = self.thisptr.numDim_
        return result

    cpdef set_num_dim_(TrajectoryData self, int numDim_):
        cdef int cpp_numDim_ = numDim_
        self.thisptr.numDim_ = cpp_numDim_

    is_stroke_ = property(get_is_stroke_, set_is_stroke_)

    cpdef get_is_stroke_(TrajectoryData self):
        cdef bool result = self.thisptr.isStroke_
        return result

    cpdef set_is_stroke_(TrajectoryData self, bool isStroke_):
        cdef bool cpp_isStroke_ = isStroke_
        self.thisptr.isStroke_ = cpp_isStroke_

    overlap_ = property(get_overlap_, set_overlap_)

    cpdef get_overlap_(TrajectoryData self):
        cdef double result = self.thisptr.overlap_
        return result

    cpdef set_overlap_(TrajectoryData self, double overlap_):
        cdef double cpp_overlap_ = overlap_
        self.thisptr.overlap_ = cpp_overlap_


    cpdef sample_trajectory_data(TrajectoryData self, TrajectoryData traj):
        cdef cpp.TrajectoryData * cpp_traj = traj.thisptr
        self.thisptr.sampleTrajectoryData(deref(cpp_traj))

    cpdef step(TrajectoryData self, double timestamp, np.ndarray[double, ndim=1] values, np.ndarray[double, ndim=1] covs):
        cdef double cpp_timestamp = timestamp
        self.thisptr.step(cpp_timestamp, &values[0], values.shape[0], &covs[0], covs.shape[0])

    cpdef imitate(TrajectoryData self, np.ndarray[double, ndim=1] sizes, np.ndarray[double, ndim=1] timestamps, np.ndarray[double, ndim=1] values):
        self.thisptr.imitate(&sizes[0], sizes.shape[0], &timestamps[0], timestamps.shape[0], &values[0], values.shape[0])

    cpdef get_values(TrajectoryData self, np.ndarray[double, ndim=1] timestamps, np.ndarray[double, ndim=1] means, np.ndarray[double, ndim=1] covars):
        self.thisptr.getValues(&timestamps[0], timestamps.shape[0], &means[0], means.shape[0], &covars[0], covars.shape[0])

    cpdef condition(TrajectoryData self, int count, np.ndarray[double, ndim=1] points):
        cdef int cpp_count = count
        self.thisptr.condition(cpp_count, &points[0], points.shape[0])


cdef class CombinedTrajectoryData:
    cdef cpp.CombinedTrajectoryData * thisptr
    cdef bool delete_thisptr

    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = True

    def __dealloc__(self):
        if self.delete_thisptr and self.thisptr != NULL:
            del self.thisptr

    def __init__(CombinedTrajectoryData self):
        self.thisptr = new cpp.CombinedTrajectoryData()

    means_ = property(get_means_, set_means_)

    cpdef get_means_(CombinedTrajectoryData self):
        cdef vector[vector[double] ] result = self.thisptr.means_
        return result

    cpdef set_means_(CombinedTrajectoryData self, object means_):
        cdef vector[vector[double] ] cpp_means_ = means_
        self.thisptr.means_ = cpp_means_

    covariances_ = property(get_covariances_, set_covariances_)

    cpdef get_covariances_(CombinedTrajectoryData self):
        cdef vector[vector[double] ] result = self.thisptr.covariances_
        return result

    cpdef set_covariances_(CombinedTrajectoryData self, object covariances_):
        cdef vector[vector[double] ] cpp_covariances_ = covariances_
        self.thisptr.covariances_ = cpp_covariances_

    activations_ = property(get_activations_, set_activations_)

    cpdef get_activations_(CombinedTrajectoryData self):
        cdef vector[vector[double] ] result = self.thisptr.activations_
        return result

    cpdef set_activations_(CombinedTrajectoryData self, object activations_):
        cdef vector[vector[double] ] cpp_activations_ = activations_
        self.thisptr.activations_ = cpp_activations_

    num_b_f_ = property(get_num_b_f_, set_num_b_f_)

    cpdef get_num_b_f_(CombinedTrajectoryData self):
        cdef int result = self.thisptr.numBF_
        return result

    cpdef set_num_b_f_(CombinedTrajectoryData self, int numBF_):
        cdef int cpp_numBF_ = numBF_
        self.thisptr.numBF_ = cpp_numBF_

    num_dim_ = property(get_num_dim_, set_num_dim_)

    cpdef get_num_dim_(CombinedTrajectoryData self):
        cdef int result = self.thisptr.numDim_
        return result

    cpdef set_num_dim_(CombinedTrajectoryData self, int numDim_):
        cdef int cpp_numDim_ = numDim_
        self.thisptr.numDim_ = cpp_numDim_

    is_stroke_ = property(get_is_stroke_, set_is_stroke_)

    cpdef get_is_stroke_(CombinedTrajectoryData self):
        cdef bool result = self.thisptr.isStroke_
        return result

    cpdef set_is_stroke_(CombinedTrajectoryData self, bool isStroke_):
        cdef bool cpp_isStroke_ = isStroke_
        self.thisptr.isStroke_ = cpp_isStroke_

    overlap_ = property(get_overlap_, set_overlap_)

    cpdef get_overlap_(CombinedTrajectoryData self):
        cdef double result = self.thisptr.overlap_
        return result

    cpdef set_overlap_(CombinedTrajectoryData self, double overlap_):
        cdef double cpp_overlap_ = overlap_
        self.thisptr.overlap_ = cpp_overlap_


    cpdef add_trajectory(CombinedTrajectoryData self, TrajectoryData trajectory, np.ndarray[double, ndim=1] activation):
        cdef cpp.TrajectoryData * cpp_trajectory = trajectory.thisptr
        self.thisptr.addTrajectory(deref(cpp_trajectory), &activation[0], activation.shape[0])

    cpdef step(CombinedTrajectoryData self, double timestamp, np.ndarray[double, ndim=1] values, np.ndarray[double, ndim=1] covs):
        cdef double cpp_timestamp = timestamp
        self.thisptr.step(cpp_timestamp, &values[0], values.shape[0], &covs[0], covs.shape[0])

    cpdef get_values(CombinedTrajectoryData self, np.ndarray[double, ndim=1] timestamps, np.ndarray[double, ndim=1] means, np.ndarray[double, ndim=1] covars):
        self.thisptr.getValues(&timestamps[0], timestamps.shape[0], &means[0], means.shape[0], &covars[0], covars.shape[0])

