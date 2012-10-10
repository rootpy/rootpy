#include <Python.h>
#include <numpy/arrayobject.h>


static PyObject *fill_hist_with_ndarray(PyObject *self, PyObject *args) {
    PyObject *input;
    PyArrayObject *array;
    double sum;
    int i, n;

    if (!PyArg_ParseTuple(args, "O", &input))
        return NULL;
    array = (PyArrayObject *)
    PyArray_ContiguousFromObject(input, PyArray_DOUBLE, 2, 2);
    if (array == NULL)
        return NULL;

    n = array->dimensions[0];
    if (n > array->dimensions[1])
        n = array->dimensions[1];
    sum = 0.;
    for (i = 0; i < n; i++)
        sum += *(double *)(array->data + i*array->strides[0] + i*array->strides[1]);
    Py_DECREF(array);
    return PyFloat_FromDouble(sum);
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
