#include <Python.h>
#include <numpy/arrayobject.h>
#include <TH1.h>


static PyObject *fill_hist_with_ndarray(PyObject *self, PyObject *args, PyObject* keywords) {
    using namespace std;
    PyObject *hist_ = NULL;
    PyObject *array_ = NULL;
    PyObject *weights_ = NULL;
    PyArrayObject *array = NULL;
    PyArrayObject *weights = NULL;
    double weight = 1.;
    unsigned int i, n;
    static const char* keywordslist[] = {"hist", "array", "weights", NULL};

    if(!PyArg_ParseTupleAndKeywords(
                args, keywords, "OO|O",
                const_cast<char **>(keywordslist),
                &hist_, &array_, &weights_)) {
        return NULL;
    }

    if(!PyCObject_Check(hist_)) {
        PyErr_SetString(PyExc_TypeError,"Unable to convert hist to PyCObject");
        return NULL;
    }
    //this is not safe so be sure to know what you are doing type check in python first
    //this is a c++ limitation because void* have no vtable so dynamic cast doesn't work
    TH1* hist = static_cast<TH1*>(PyCObject_AsVoidPtr(hist_));

    if(!hist) {
        PyErr_SetString(PyExc_TypeError,"Unable to convert hist to TH1*");
        return NULL;
    }

    array = (PyArrayObject *) PyArray_ContiguousFromAny(
            array_, PyArray_DOUBLE, 1, 1);
    if (array == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "Unable to convert object to array");
        return NULL;
    }

    if (weights_) {
        weights = (PyArrayObject *) PyArray_ContiguousFromAny(
                weights_, PyArray_DOUBLE, 1, 1);
        if (weights == NULL) {
            PyErr_SetString(PyExc_TypeError,
                    "Unable to convert object to array");
            Py_DECREF(array);
            return NULL;
        }
    }

    n = array->dimensions[0];
    // weighted fill
    if (weights) {
        if (n != weights->dimensions[0]) {
            PyErr_SetString(PyExc_ValueError,
                    "array and weights must have the same length");
            Py_DECREF(weights);
            Py_DECREF(array);
            return NULL;
        }
        for (i = 0; i < n; ++i) {
            hist->Fill(*(double *)(array->data + i * array->strides[0]),
                       *(double *)(weights->data + i * weights->strides[0]));
        }
        Py_DECREF(weights);
    } else {
        for (i = 0; i < n; ++i) {
            hist->Fill(*(double *)(array->data + i * array->strides[0]));
        }
    }
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
