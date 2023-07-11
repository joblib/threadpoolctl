int NUM_THREADS = 42;
int* NUM_THREADS_p = &NUM_THREADS;


int mylib_get_num_threads(){
    return *NUM_THREADS_p;
}


void mylib_set_num_threads(int num_threads){
    *NUM_THREADS_p = num_threads;
}


char* mylib_get_version(){
    return "2.0";
}
