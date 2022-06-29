#include <time.h>
#include <stdint.h>

double benchmark(
    int32_t *tensorIn,
    int32_t *kernel,
    int32_t *tensorOut,
    int N,
    int IH, int IW, int IC, int OC,
    int KH, int KW
)
{
    int num_iter = 500;

    struct timespec start, end;
    double total_time = 0;

    for (int eval_iter=0;eval_iter<num_iter;eval_iter++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        inference(tensorIn, kernel, tensorOut, N, IH, IW, IC, OC, KH, KW);
        clock_gettime(CLOCK_MONOTONIC, &end);

        total_time += (double)(end.tv_sec - start.tv_sec)*1000000 + (double)(end.tv_nsec - start.tv_nsec)/1000.0;
    }

    return total_time / (double)(num_iter);
}

int inference(
    int32_t *tensorIn,
    int32_t *kernel,
    int32_t *tensorOut,
    int N,
    int IH, int IW, int IC, int OC,
    int KH, int KW
)
{
    for(int n = 0; n < N; ++n) {
        for(int h = -1 * (KH / 2); h < IH - (KH / 2); ++h) {
            for(int w = -1 * (KW / 2); w < IW - (KW / 2); ++w) {
   	            for(int oc = 0; oc < OC; ++oc) {
   			        int32_t val = 0;

                    int nhw = n * IH * IW * IC + h * IW * IC + w * IC;

   		            for(int kw=0; kw<KW; ++kw) {
   				        for(int kh=0; kh<KH; ++kh) {
 				    		for(int ic = 0; ic < IC; ++ic) {
 				    		    if((h+kh>=0) && (h+kh<IH) && (w+kw>=0) && (w+kw<IW)) {
                                    val += tensorIn[n * IH*IW*IC+h * IW*IC+w * IC + kh*IW*IC + kw*IC + ic]
                                        * kernel[oc * KH*KW*IC + kh*KW*IC + kw*IC + ic];
                                }
 	                        }
                        }
                    }
                    tensorOut[n * IH*IW*OC + (h+KH/2)*IW*OC + (w+KW/2)*OC + oc] = val;                    
                }
            }
        }
    }


    return 0;
    /* Code Ends Here */
}