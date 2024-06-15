   #include <Python.h>
   #include <stdlib.h>
   #include <time.h>

   static PyObject* py_generate_random_graph(PyObject* self, PyObject* args) {
       int num_people;
       const char* file_path;
       unsigned int seed;

       if (!PyArg_ParseTuple(args, "isi", &num_people, &file_path, &seed)) {
           return NULL;
       }

       srand(seed);
       FILE *file = fopen(file_path, "w");
       if (!file) {
           PyErr_SetString(PyExc_IOError, "Failed to open file");
           return NULL;
       }

       for (int i = 0; i < num_people; i++) {
           for (int j = 0; j < num_people; j++) {
               if (i == j) {
                   fprintf(file, "0");
               } else {
                   float strength = (float)rand() / RAND_MAX;
                   if (strength < 0.1) {
                       fprintf(file, "1");
                   } else if (strength < 0.4) {
                       fprintf(file, "1");
                   } else {
                       fprintf(file, "0");
                   }
               }
               if (j < num_people - 1) {
                   fprintf(file, ",");
               }
           }
           fprintf(file, "\n");
       }

       fclose(file);
       Py_RETURN_NONE;
   }

   static PyObject* py_generate_weight_matrix(PyObject* self, PyObject* args) {
       int num_nodes, weight_min, weight_max;
       const char* file_path;
       unsigned int seed;

       if (!PyArg_ParseTuple(args, "iiisi", &num_nodes, &weight_min, &weight_max, &file_path, &seed)) {
           return NULL;
       }

       srand(seed);
       FILE *file = fopen(file_path, "w");
       if (!file) {
           PyErr_SetString(PyExc_IOError, "Failed to open file");
           return NULL;
       }

       for (int i = 0; i < num_nodes; i++) {
           for (int j = 0; j < num_nodes; j++) {
               if (i == j) {
                   fprintf(file, "0");
               } else {
                   int weight = weight_min + rand() % (weight_max - weight_min + 1);
                   fprintf(file, "%d", weight);
               }
               if (j < num_nodes - 1) {
                   fprintf(file, ",");
               }
           }
           fprintf(file, "\n");
       }

       fclose(file);
       Py_RETURN_NONE;
   }

   static PyMethodDef VishGraphsMethods[] = {
       {"generate_random_graph", py_generate_random_graph, METH_VARARGS, "Generate a random graph"},
       {"generate_weight_matrix", py_generate_weight_matrix, METH_VARARGS, "Generate a weight matrix"},
       {NULL, NULL, 0, NULL}
   };

   static struct PyModuleDef vishgraphsmodule = {
       PyModuleDef_HEAD_INIT,
       "vish_graphs",
       NULL,
       -1,
       VishGraphsMethods
   };

   PyMODINIT_FUNC PyInit_vish_graphs(void) {
       return PyModule_Create(&vishgraphsmodule);
   }