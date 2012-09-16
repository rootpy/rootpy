// A specialized handler to allow python functions to be called by
// the TCanvas::ProcessedEvent

#include <TPython.h>
#include <TPyDispatcher.h>

class TPyDispatcherProcessedEvent : public TPyDispatcher {
public:
    TPyDispatcherProcessedEvent(PyObject* callable) : TPyDispatcher(callable) {}

    PyObject* Dispatch(int p1, int p2, int p3, void* p4) {
        if (!p4) return NULL;
        PyObject* p4_aspyobj = TPython::ObjectProxy_FromVoidPtr(p4,
            reinterpret_cast<TObject*>(p4)->ClassName());
        PyObject* result = DispatchVA("lllO", p1, p2, p3, p4_aspyobj);
        return result;
    }

    ClassDef(TPyDispatcherProcessedEvent, 0);
};

ClassImp(TPyDispatcherProcessedEvent);
