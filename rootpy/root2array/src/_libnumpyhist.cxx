#include <Python.h>
#include <numpy/arrayobject.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>


static PyObject *
fill_hist_with_ndarray(PyObject *self, PyObject *args, PyObject* keywords) {

    using namespace std;
    PyObject *hist_ = NULL;
    PyObject *array_ = NULL;
    PyObject *weights_ = NULL;
    PyArrayObject *array = NULL;
    PyArrayObject *weights = NULL;
    TH1* hist = NULL;
    TH2* hist2d = NULL;
    TH3* hist3d = NULL;
    unsigned int dim = 1;
    unsigned int array_depth = 1;
    unsigned int i, n, k;
    char *array_data;
    char *weights_data = NULL;
    npy_intp array_stride_n;
    npy_intp array_stride_k = 1;
    npy_intp weights_stride = 1;
    static const char* keywordslist[] = {
        "hist",
        "dim",
        "array",
        "weights",
        NULL};

    if(!PyArg_ParseTupleAndKeywords(
                args, keywords, "OIO|O",
                const_cast<char **>(keywordslist),
                &hist_, &dim, &array_, &weights_)) {
        return NULL;
    }

    if(!PyCObject_Check(hist_)) {
        PyErr_SetString(PyExc_TypeError,"Unable to convert hist to PyCObject");
        return NULL;
    }
    //this is not safe so be sure to know what you are doing type check in python first
    //this is a c++ limitation because void* have no vtable so dynamic cast doesn't work
    if (dim == 1) {
        hist = static_cast<TH1*>(PyCObject_AsVoidPtr(hist_));
        if(!hist) {
            PyErr_SetString(PyExc_TypeError,"Unable to convert hist to TH1*");
            return NULL;
        }
    } else if (dim == 2) {
        hist2d = static_cast<TH2*>(PyCObject_AsVoidPtr(hist_));
        if(!hist2d) {
            PyErr_SetString(PyExc_TypeError,"Unable to convert hist to TH2*");
            return NULL;
        }
    } else if (dim == 3) {
        hist3d = static_cast<TH3*>(PyCObject_AsVoidPtr(hist_));
        if(!hist3d) {
            PyErr_SetString(PyExc_TypeError,"Unable to convert hist to TH3*");
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_ValueError,
                "dim must not be greater than 3");
        return NULL;
    }

    if (dim > 1)
        array_depth = 2;

    array = (PyArrayObject *) PyArray_ContiguousFromAny(
            array_, PyArray_DOUBLE, array_depth, array_depth);
    if (array == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "Unable to convert object to array");
        return NULL;
    }

    if (dim > 1) {
        k = array->dimensions[1];
        if (k != dim) {
            PyErr_SetString(PyExc_ValueError,
                "length of the second dimension must equal the dimension of the histogram");
            Py_DECREF(array);
            return NULL;
        }
        array_stride_k = array->strides[1];
    }

    n = array->dimensions[0];
    array_stride_n = array->strides[0];
    array_data = array->data;

    if (weights_) {
        weights = (PyArrayObject *) PyArray_ContiguousFromAny(
                weights_, PyArray_DOUBLE, 1, 1);
        if (weights == NULL) {
            PyErr_SetString(PyExc_TypeError,
                    "Unable to convert object to array");
            Py_DECREF(array);
            return NULL;
        }
        if (n != weights->dimensions[0]) {
            PyErr_SetString(PyExc_ValueError,
                    "array and weights must have the same length");
            Py_DECREF(weights);
            Py_DECREF(array);
            return NULL;
        }
        weights_data = weights->data;
        weights_stride = weights->strides[0];
    }

    if (dim == 1) {
        // weighted fill
        if (weights) {
            for (i = 0; i < n; ++i) {
                hist->Fill(
                        *(double *)(array_data + i * array_stride_n),
                        *(double *)(weights_data + i * weights_stride));
            }
        } else {
            // unweighted fill
            for (i = 0; i < n; ++i) {
                hist->Fill(*(double *)(array_data + i * array_stride_n));
            }
        }
    } else if (dim == 2) {
        // weighted fill
        if (weights) {
            for (i = 0; i < n; ++i) {
                hist2d->Fill(
                        *(double *)(array_data + i * array_stride_n),
                        *(double *)(array_data + i * array_stride_n + array_stride_k),
                        *(double *)(weights_data + i * weights_stride));
            }
        } else {
            // unweighted fill
            for (i = 0; i < n; ++i) {
                hist2d->Fill(
                        *(double *)(array_data + i * array_stride_n),
                        *(double *)(array_data + i * array_stride_n + array_stride_k));
            }
        }
    } else if (dim == 3) {
        // weighted fill
        if (weights) {
            for (i = 0; i < n; ++i) {
                hist3d->Fill(
                        *(double *)(array_data + i * array_stride_n),
                        *(double *)(array_data + i * array_stride_n + array_stride_k),
                        *(double *)(array_data + i * array_stride_n + 2 * array_stride_k),
                        *(double *)(weights_data + i * weights_stride));
            }
        } else {
            // unweighted fill
            for (i = 0; i < n; ++i) {
                hist3d->Fill(
                        *(double *)(array_data + i * array_stride_n),
                        *(double *)(array_data + i * array_stride_n + array_stride_k),
                        *(double *)(array_data + i * array_stride_n + 2 * array_stride_k));
            }
        }
    }
    if (weights)
        Py_DECREF(weights);
    Py_DECREF(array);
    Py_RETURN_NONE;
}


static PyMethodDef methods[] = {
    {"fill_hist_with_ndarray",  (PyCFunction)fill_hist_with_ndarray,
     METH_VARARGS|METH_KEYWORDS,
     ""},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

void cleanup(){
    //do nothing
}

PyMODINIT_FUNC
init_libnumpyhist(void)
{
    import_array();
    (void) Py_InitModule("_libnumpyhist", methods);
}
